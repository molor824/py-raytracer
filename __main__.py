import pygame
import multiprocessing as mp
import multiprocessing.shared_memory as shm

from raytracing import *

WIDTH = 800
HEIGHT = 600

BUF_WIDTH = WIDTH // 2
BUF_HEIGHT = HEIGHT // 2

FPS = 30
DELTA = 1.0 / FPS

LOOK_SPEED = 0.5
MOVE_SPEED = 100.0

def write_mem(mem: shm.SharedMemory, params):
    pickled = pickle.dumps(params)
    mem.buf[0:len(pickled)] = pickled

def rot_x(radian):
    c = math.cos(radian)
    s = math.sin(radian)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0, s, c]])

def rot_y(radian: float):
    c = math.cos(radian)
    s = math.sin(radian)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])

def rot_z(radian: float):
    c = math.cos(radian)
    s = math.sin(radian)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])

def scale(scale: tuple):
    scale = np.array(scale)
    return np.identity(scale.shape[0]) * scale[:, np.newaxis]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    render_surface = pygame.Surface((BUF_WIDTH, BUF_HEIGHT))

    pixel_mem = shm.SharedMemory(create=True, size=BUF_WIDTH * BUF_HEIGHT * 3)
    params_mem = shm.SharedMemory(create=True, size=10000)

    parameters = Parameters((BUF_WIDTH, BUF_HEIGHT), Transform(translation=np.array((0.0, 0.0, -300.0))), [
        Sphere(20, Transform(translation=np.array((0.0, -40.0, 0.0))), Material((0, 255, 255))),
        Cube(
            (30.0,) * 3,
            Transform(spatial=rot_y(np.deg2rad(-30.0)), translation=np.array((0.0, 0.0, 0.0))),
            Material((0, 255, 0))
        ),
        Cube(
            (200.0, 20.0, 200.0),
            Transform(spatial=rot_y(np.deg2rad(45.0)), translation=np.array((0.0, 40.0, 0.0))),
            Material((255, 0, 255))
        )
    ], [
        Light(np.array([0.7, 1.0, 0.3])),
    ])
    write_mem(params_mem, parameters)

    cpu_count = mp.cpu_count()
    update_events = [mp.Event() for _ in range(cpu_count)]
    processes = [mp.Process(target=raytrace, args=(cpu_count, i, pixel_mem.name, params_mem.name, update_events[i])) for i in range(cpu_count)]
    for p in processes: p.start()

    camera_y = 0.0
    camera_x = 0.0

    running = True
    while running:
        clock.tick(FPS)

        keys = pygame.key.get_pressed()
        look_direction = [0, 0]
        if keys[pygame.K_RIGHT]:
            look_direction[0] += 1
        if keys[pygame.K_LEFT]:
            look_direction[0] -= 1
        if keys[pygame.K_UP]:
            look_direction[1] -= 1
        if keys[pygame.K_DOWN]:
            look_direction[1] += 1

        move_direction = [0, 0]
        if keys[pygame.K_w]:
            move_direction[1] += 1
        if keys[pygame.K_s]:
            move_direction[1] -= 1
        if keys[pygame.K_d]:
            move_direction[0] += 1
        if keys[pygame.K_a]:
            move_direction[0] -= 1

        updated = False

        if look_direction[0] != 0 or look_direction[1] != 0:
            camera_x += -look_direction[1] * (LOOK_SPEED * DELTA)
            camera_y += look_direction[0] * (LOOK_SPEED * DELTA)
            camera_x = np.clip(camera_x, -np.pi / 2, np.pi / 2)
            parameters.transform.spatial = np.dot(rot_y(camera_y), rot_x(camera_x))
            updated = True

        if move_direction[0] != 0 or move_direction[1] != 0:
            direction = np.dot(parameters.transform.spatial, np.array([move_direction[0], 0, move_direction[1]], dtype=np.float32))
            parameters.transform.translation += direction * (MOVE_SPEED * DELTA)
            updated = True

        if updated:
            write_mem(params_mem, parameters)
            for event in update_events:
                event.set()

        array = np.resize(np.frombuffer(pixel_mem.buf, dtype=np.uint8), (BUF_WIDTH, BUF_HEIGHT, 3))

        pygame.surfarray.blit_array(render_surface, array)
        pygame.transform.scale(render_surface, (WIDTH, HEIGHT), screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    if processes is not None:
        for p in processes:
            p.kill()
            p.join()
            p.close()
    pixel_mem.unlink()
    params_mem.unlink()
    pixel_mem.close()
    params_mem.close()
    pygame.quit()

if __name__ == '__main__':
    main()
