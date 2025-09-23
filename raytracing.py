import multiprocessing.shared_memory as shm
import struct
import pickle
import numpy as np
import math

class Object:
    def __init__(self, spatial: np.matrix, translation: np.ndarray):
        self.spatial = spatial
        self.translation = translation

class RayIntersection:
    def __init__(self, distance: float, normal: np.ndarray):
        self.distance = distance
        self.normal = normal

class Circle(Object):
    def __init__(self, spatial: np.matrix, translation: np.ndarray, radius: float):
        super().__init__(spatial, translation)
        self.radius = radius
    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        world_origin = origin
        inv_spatial = np.linalg.inv(self.spatial)
        origin = np.matmul(inv_spatial, origin - self.translation)
        direction = np.matmul(inv_spatial, direction)
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
        world_point = np.matmul(self.spatial, hit_point) + self.translation
        world_normal = np.matmul(np.linalg.inv(np.transpose(self.spatial)), hit_normal)
        world_normal /= np.linalg.norm(world_normal)
        dist = np.linalg.norm(world_origin - world_point)
        return RayIntersection(dist, world_normal)

class Parameters:
    def __init__(self, position: np.ndarray, fov = 1.0):
        self.position = position
        self.fov = fov
        self.objects: list[Object] = []

def load_buffer(buffer):
    size = struct.unpack("i", buffer[0:4])[0]
    return pickle.loads(buffer[4:size + 4])

def raytrace(size: np.ndarray, process_count: int, process_index: int, pixel_buffer_name: str, params_buffer_name: str, quit_event):
    pixel_mem = shm.SharedMemory(pixel_buffer_name)
    params_mem = shm.SharedMemory(params_buffer_name)

    center = size / 2
    
    while True:
        for y in range(process_index, size[1], process_count):
            for x in range(size[0]):
                if quit_event.is_set(): return

                params: Parameters = load_buffer(params_mem.buf)
                origin = params.position
                direction = np.array((x + 0.5 - center[0], (y + 0.5 - center[1]) * params.fov, size[1]), dtype=np.float64)
                start = (x * size[1] + y) * 3

                result = min((obj.intersect(origin, direction) for obj in params.objects), key=lambda hit: hit.distance if hit != None else math.inf)
                
                if result != None:
                    pixel_mem.buf[start:start+3] = np.astype(result.normal * 127 + 127, np.uint8).tobytes()
                else:
                    pixel_mem.buf[start:start+3] = bytes((120, 120, 120))
