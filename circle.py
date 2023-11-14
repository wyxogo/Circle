import taichi as ti
import taichi.math as tm

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

ti.init(arch=ti.cpu,
        random_seed=42)

min_distance = 12
max_distance = 18  

grid_n = 1200
width = grid_n
height = grid_n
res = (width, height)
dx= 1
inv_dx = 1/dx
radius = 5
desired_samples = 1000000
grid = ti.field(dtype=int, shape=res)
samples = ti.Vector.field(2, float, shape=desired_samples)
grid.fill(-1)

@ti.func
def check_collision_test(p, index, d):
    x, y = index
    collision = False
    s = 1.5
    for i in range(max(0, x - int(s*d)), min(width-0, x + int(s*d))): 
        for j in range(max(0, y - int(s*d)), min(height-0, y + int(s*d))): 
            if grid[i, j] != -1:
                q = samples[grid[i, j]]
                if (q - p).norm() < min_distance:
                    collision = True
    return collision

@ti.kernel
def poisson_disk_sample_test(desired_samples: int, r: float, theta_list: ti.types.ndarray()) -> int:
    samples[0] = tm.vec2(width / 2)
    d_index = 0
    grid[int(width / 2), int(width / 2)] = 0
    head, tail = 0, 1
    angular_step = tm.pi * 2 / theta_list.shape[0]
    while head < tail and head < desired_samples:
        source_x = samples[head]
        head += 1
        for theta_ in theta_list:
            theta = theta_*angular_step
            unit_offset = tm.vec2(tm.cos(theta), tm.sin(theta)).normalized()
            offset = unit_offset * (r+ti.random()*6)
            new_x = source_x + offset
            new_index = int(new_x * inv_dx)
            if radius <= new_x[0] <= width-radius and radius <= new_x[1] <= height-radius:
                collision = check_collision_test(new_x, new_index, r)
                if not collision and tail < desired_samples:
                    samples[tail] = new_x
                    grid[new_index] = tail
                    tail += 1
        d_index += 1
    return tail

time1 = time.time()
r = min_distance
theta_list = np.array(range(500))
np.random.shuffle(theta_list)
num_samples = poisson_disk_sample_test(desired_samples, r, theta_list)
time2 = time.time()
print(f"Circle Generation: {time2 - time1} s")
samples_np = samples.to_numpy()[:num_samples]
print(f"Samples Shape: {samples_np.shape}")

import ezdxf
time1 = time.time()
# Create a new DXF document.
doc = ezdxf.new(dxfversion='R2018')  # Use the appropriate DXF version for your application.
msp = doc.modelspace()
# Add circles to the DXF document.
for sample in samples_np:
    msp.add_circle(center=(sample[0], sample[1]), radius=5)
# Save the DXF document.
import os
if not os.path.exists('output'):
    os.makedirs('output')
doc.saveas('./output/samples_np.dxf')
time2 = time.time()
print(f"DXF save: {time2 - time1}")


def im_show(distances, centers, cnt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(80, 40))
    ax1.hist(distances, bins=np.linspace(min_distance, max_distance, 20))
    ax1.set_title(f'Distance Distribution, {cnt} Centers')
    
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect('equal')
    
    for center in centers:
        circle = plt.Circle(center, radius, fill=False)
        ax2.add_artist(circle)
    ax2.set_title('Circle Visualization')
    plt.savefig(f"./output/samples_np.png")
    plt.close(fig)

time1 = time.time()
tree = KDTree(samples_np)
dist, _ = tree.query(samples_np, k=10)
distances = dist[:,1:].flatten()
im_show(distances, samples_np, len(samples_np))
time2 = time.time()
print(f"Plot save: {time2 - time1} s")
print(f"Min Distance: {np.amin(distances)}")
per_cnt = np.sum((distances >= min_distance) & (distances <= max_distance))/len(samples_np)
print(f'Avg number of Distance: {per_cnt}')