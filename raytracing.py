from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event
import numpy as np
import math
import pickle

class Transform:
    def __init__(self, spatial = np.identity(3), translation = np.array([0.0, 0.0, 0.0])):
        self.spatial = spatial
        self.translation = translation
    
    def inv(self):
        inv_spatial = np.linalg.inv(self.spatial)
        return Transform(inv_spatial, np.matmul(inv_spatial, -self.translation))
    
    def mul_point(self, point: np.ndarray):
        return np.matmul(self.spatial, point) + self.translation
    def mul_vector(self, vector: np.ndarray):
        return np.matmul(self.spatial, vector)
    def mul_normal(self, normal: np.ndarray):
        normal = np.matmul(np.linalg.inv(np.transpose(self.spatial)), normal)
        return normal / np.linalg.norm(normal)

class Object:
    def __init__(self, transform: Transform):
        self.transform = transform

class RayIntersection:
    def __init__(self, distance: float, normal: np.ndarray):
        self.distance = distance
        self.normal = normal

class Circle(Object):
    def __init__(self, transform: Transform, radius: float):
        super().__init__(transform)
        self.radius = radius
    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        world_origin = origin
        inv_transform = self.transform.inv()
        origin = inv_transform.mul_point(origin)
        direction = inv_transform.mul_vector(direction)
        direction /= np.linalg.norm(direction)
        projected_distance = -np.dot(origin, direction)
        projected_point = origin + direction * projected_distance
        closest_distance = np.linalg.norm(projected_point)

        if closest_distance > self.radius:
            return

        hit_dist_from_circle = math.sqrt(self.radius * self.radius - closest_distance * closest_distance)
        hit_dist = projected_distance - hit_dist_from_circle

        if hit_dist < 0.0:
            return

        hit_point = origin + direction * hit_dist
        hit_normal = hit_point
        world_point = self.transform.mul_point(hit_point)
        return RayIntersection(np.linalg.norm(world_origin - world_point), self.transform.mul_normal(hit_normal))

class Parameters:
    def __init__(self, size: np.ndarray, transform: Transform, objects: list):
        self.size = size
        self.transform = transform
        self.objects = objects

def raytrace(process_count: int, process_index: int, pixel_buffer_name: str, params_buffer_name: str, update_event = Event()):
    pixel_mem = SharedMemory(pixel_buffer_name)
    params_mem = SharedMemory(params_buffer_name)
    print(f"Process index {process_index}")

    while True:
        params: Parameters = pickle.loads(params_mem.buf)
        size = params.size
        transform = params.transform
        objects = params.objects
        center = size / 2

        xs = np.arange(size[0])
        ys = np.arange(process_index, size[1], process_count)
        pixel_indices = (xs[:, np.newaxis] * size[1] + ys).ravel()
        np.random.shuffle(pixel_indices)

        for i in pixel_indices:
            if update_event.is_set():
                break

            x = i // size[1]
            y = i % size[1]
            origin = transform.translation
            direction = transform.mul_vector(np.array((x + 0.5 - center[0], (y + 0.5 - center[1]), size[1])))
            start = i * 3

            result = min((obj.intersect(origin, direction) for obj in objects), key=lambda hit: hit.distance if hit != None else math.inf)

            if result != None:
                pixel_mem.buf[start:start+3] = np.astype(result.normal * 127 + 127, np.uint8).tobytes()
            else:
                pixel_mem.buf[start:start+3] = bytes((120, 120, 120))

        update_event.wait()
        update_event.clear()
