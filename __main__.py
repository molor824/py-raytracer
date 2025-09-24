import pygame
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import pickle

from raytracing import *

WIDTH = 800
HEIGHT = 600

BUF_WIDTH = WIDTH // 2
BUF_HEIGHT = HEIGHT // 2

FPS = 30
DELTA = 1.0 / FPS

def write_mem(mem: shm.SharedMemory, params):
    pickled = pickle.dumps(params)
    mem.buf[0:len(pickled)] = pickled

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    render_surface = pygame.Surface((BUF_WIDTH, BUF_HEIGHT))

    pixel_mem = shm.SharedMemory(create=True, size=BUF_WIDTH * BUF_HEIGHT * 3)
    params_mem = shm.SharedMemory(create=True, size=10000)

    parameters = Parameters((BUF_WIDTH, BUF_HEIGHT), Transform(), [
        Circle(Transform(translation=np.array((0.0, 0.0, 200.0))), 50.0),
        Circle(Transform(translation=np.array((0.0, 30.0, 500.0))), 100.0),
        Cube(Transform(translation=np.array((20.0, 0.0, 300.0))), (100.0, 60.0, 120.0))
    ])
    write_mem(params_mem, parameters)

    cpu_count = mp.cpu_count()
    update_events = [mp.Event() for _ in range(cpu_count)]
    processes = [mp.Process(target=raytrace, args=(cpu_count, i, pixel_mem.name, params_mem.name, update_events[i])) for i in range(cpu_count)]
    for p in processes: p.start()

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
        
        updated = False
        
        if direction[0] != 0.0 or direction[1] != 0.0:
            parameters.transform.translation += np.array(direction) * (DELTA * 100.0)
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

    if processes != None:
        for p in processes:
            p.kill()
            p.join()
            p.close()
    pixel_mem.unlink()
    params_mem.unlink()
    pixel_mem.close()
    params_mem.close()
    pygame.quit()

if __name__ == "__main__":
    main()
