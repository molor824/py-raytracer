import pygame
import raytracing
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import pickle
import struct
import numpy as np

WIDTH = 800
HEIGHT = 600

BUF_WIDTH = WIDTH // 2
BUF_HEIGHT = HEIGHT // 2

FPS = 60
DELTA = 1.0 / FPS

OBJ_MEM_CAPACITY = 1024 * 1024

def write_buffer(buffer, value):
    pickled = pickle.dumps(value)
    buffer[0:4] = struct.pack("i", len(pickled))
    buffer[4:len(pickled) + 4] = pickled

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    render_surface = pygame.Surface((BUF_WIDTH, BUF_HEIGHT))

    pixel_mem = shm.SharedMemory(create=True, size=BUF_WIDTH * BUF_HEIGHT * 3)

    params_mem = shm.SharedMemory(create=True, size=OBJ_MEM_CAPACITY)
    params = raytracing.Parameters(np.array((0.0, 0.0, 0.0)))
    params.objects.append(raytracing.Circle(np.array((0.0, 0.0, 200.0)), 50.0))
    params.objects.append(raytracing.Circle(np.array((0.0, 30.0, 500.0)), 100.0))
    write_buffer(params_mem.buf, params)

    quit_event = mp.Event()

    cpu_count = mp.cpu_count() - 1
    processes = [mp.Process(
        target=raytracing.raytrace,
        args=(np.array((BUF_WIDTH, BUF_HEIGHT)), cpu_count, i, pixel_mem.name, params_mem.name, quit_event)
    ) for i in range(cpu_count)]

    for process in processes:
        process.start()

    running = True
    while running:
        clock.tick(FPS)

        keys = pygame.key.get_pressed()
        direction = [0.0, 0.0, 0.0]
        if keys[pygame.K_RIGHT]:
            direction[0] += 1.0
        if keys[pygame.K_LEFT]:
            direction[0] -= 1.0
        if keys[pygame.K_UP]:
            direction[1] -= 1.0
        if keys[pygame.K_DOWN]:
            direction[1] += 1.0
        
        params.position = params.position + np.array(direction) * (DELTA * 40.0)
        write_buffer(params_mem.buf, params)

        array = np.resize(np.frombuffer(pixel_mem.buf, dtype=np.uint8), (BUF_WIDTH, BUF_HEIGHT, 3))

        pygame.surfarray.blit_array(render_surface, array)
        pygame.transform.scale(render_surface, (WIDTH, HEIGHT), screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    quit_event.set()

    for process in processes:
        process.join()

    pixel_mem.unlink()
    params_mem.unlink()
    pygame.quit()

if __name__ == "__main__":
    main()
