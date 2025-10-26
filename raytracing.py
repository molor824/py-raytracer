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
        return Transform(inv_spatial, np.dot(inv_spatial, -self.translation))
    
    def mul_point(self, point: np.ndarray):
        return np.dot(self.spatial, point) + self.translation
    def mul_vector(self, vector: np.ndarray):
        return np.dot(self.spatial, vector)
    def mul_normal(self, normal: np.ndarray):
        normal = np.dot(np.linalg.inv(np.transpose(self.spatial)), normal)
        return normal / np.linalg.norm(normal)

class Material:
    def __init__(self, color = (255, 255, 255), reflection = 0.0):
        self.color = color
        self.reflection = reflection

class Object:
    def __init__(self, transform: Transform, material: Material):
        self.transform = transform
        self.material = material

class RayIntersection:
    def __init__(self, distance: float, normal: np.ndarray, object: Object):
        self.distance = distance
        self.normal = normal
        self.object = object

class Sphere(Object):
    def __init__(self, radius: float, transform: Transform, material: Material):
        super().__init__(transform, material)
        self.radius = radius
    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        inv_transform = self.transform.inv()
        origin = inv_transform.mul_point(origin)
        direction = inv_transform.mul_vector(direction)
        direction /= np.linalg.norm(direction)
        projected_distance = -np.dot(origin, direction)
        projected_point = origin + direction * projected_distance
        closest_distance = np.linalg.norm(projected_point)

        if closest_distance > self.radius:
            return None

        hit_dist_from_circle = math.sqrt(self.radius * self.radius - closest_distance * closest_distance)
        hit_dist = projected_distance - hit_dist_from_circle

        if hit_dist < 0.0:
            return None

        hit_normal = origin + direction * hit_dist
        return RayIntersection(np.linalg.norm(self.transform.mul_vector(direction * hit_dist)), self.transform.mul_normal(hit_normal), self)

NORMALS0 = -np.identity(3)
NORMALS1 = np.identity(3)
NORMALS = np.append(NORMALS0, NORMALS1, 0)
MARGIN = 0.001 # allow for floating point inaccuracy
class Cube(Object):
    def __init__(self, size: tuple[float, float, float], transform: Transform, material: Material):
        super().__init__(transform, material)
        self.size = size
    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        inv_transform = self.transform.inv()
        origin = inv_transform.mul_point(origin)
        direction = inv_transform.mul_vector(direction)
        direction /= np.linalg.norm(direction)
        size = np.array(self.size)
        half_size = size * 0.5
        # calculate distance from the cube planes
        # calculation for each axis plane goes something like this
        # dist0 = -(size[axis] * 0.5 + origin[axis]) / direction[axis]
        # dist1 = (size[axis] * 0.5 - origin[axis]) / direction[axis]
        # this is the distance that is needed to intersect each plane
        dists0 = -(half_size + origin) / direction
        dists1 = (half_size - origin) / direction
        dists = np.append(dists0, dists1)
        # for each distances, we need to calculate the point at the plane
        # intersect[axis] = origin + direction * dist[axis]
        intersections = origin + direction * dists[:,np.newaxis]
        # now we need to mask out the intersections that go beyond the cube's boundary
        conditions = np.all((intersections >= -half_size - MARGIN) & (intersections <= half_size + MARGIN), 1)
        # set invalid distances with inf so the min function pretty much ignores over it
        valid_dists = np.where(conditions, dists, np.inf)
        # find min dist and normal
        min_dist_index = np.argmin(valid_dists)
        if valid_dists[min_dist_index] == np.inf or valid_dists[min_dist_index] < 0.0:
            # no valid distances found
            return None
        # finalized results
        dist = valid_dists[min_dist_index]
        normal = NORMALS[min_dist_index]
        return RayIntersection(np.linalg.norm(self.transform.mul_vector(direction * dist)), self.transform.mul_normal(normal), self)

class Light:
    def __init__(self, direction: np.ndarray, color = (255, 255, 255), intensity = 1.0):
        self.direction = direction
        self.color = color
        self.intensity = intensity

class Parameters:
    def __init__(self, size: tuple[int, int], transform: Transform, objects: list, lights=None, base_light=(40, 40, 40)):
        self.size = size
        self.transform = transform
        self.objects = objects
        self.lights = lights if lights is not None else []
        self.base_light = base_light

def raytrace(process_count: int, process_index: int, pixel_buffer_name: str, params_buffer_name: str, update_event = Event()):
    pixel_mem = SharedMemory(pixel_buffer_name)
    params_mem = SharedMemory(params_buffer_name)
    print(f"Process index {process_index}")

    while True:
        params: Parameters = pickle.loads(params_mem.buf)
        w, h = params.size
        transform = params.transform
        objects = params.objects
        lights = params.lights
        base_light = np.array(params.base_light) / 255.0
        cx, cy = w / 2, h / 2

        xs = np.arange(w)
        ys = np.arange(process_index, h, process_count)
        pixel_indices = (xs[:, np.newaxis] * h + ys).ravel()
        np.random.shuffle(pixel_indices)

        xs = pixel_indices // h
        ys = pixel_indices % h
        origin = transform.translation
        directions = np.transpose(transform.mul_vector(np.array([xs + 0.5 - cx, ys + 0.5 - cy, np.repeat(h, xs.shape[0])])))

        for i, direction in zip(pixel_indices, directions):
            if update_event.is_set():
                break

            start = i * 3
            result: RayIntersection = min((obj.intersect(origin, direction) for obj in objects), key=lambda hit: hit.distance if hit is not None else math.inf)

            if result is not None:
                mat_color = np.array(result.object.material.color) / 255.0
                min_light = mat_color * base_light
                out_color = np.zeros_like(mat_color)

                next_origin = origin + direction / np.linalg.norm(direction) * result.distance

                for l in lights:
                    light_norm = -l.direction / np.linalg.norm(l.direction)
                    hit = any((obj.intersect(next_origin, light_norm) for obj in objects if obj is not result.object))
                    if hit:
                        continue

                    diffuse_color = np.dot(light_norm, result.normal) * mat_color * l.intensity
                    out_color = out_color + diffuse_color

                pixel_mem.buf[start:start+3] = np.astype(np.clip(out_color, min_light, 1) * 255, np.uint8).tobytes()
            else:
                pixel_mem.buf[start:start+3] = bytes((120, 120, 120))

        update_event.wait()
        update_event.clear()
