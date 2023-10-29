import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
# import collections
from PIL import Image
import io

width = 0.25  
height = 0.25  
radius = 5e-3  
min_distance = 12e-3  
max_distance = 18e-3  

centers = [
            # (width/2-max_distance, height/2-max_distance),
            # (width/2+max_distance, height/2+max_distance),
            # (width/2-max_distance, height/2+max_distance),
            # (width/2+max_distance, height/2-max_distance),
            (radius, radius), 
            (width-radius, height-radius),
            (width-radius, radius),
            (radius, height-radius),
            # (width/2, height/2),
            (width*3/4, height/2),
            (width/2, height/4),
            (width/4, height/4),
            (width/2, height*3/4),
            # (width/2, radius),
            # (width-radius, height/2),
            # (radius, height/2),
            # (width/2, height-radius),

            ]
tree = KDTree(centers)

def im_show(distances, centers, cnt, curent_var):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.hist(distances, bins=40, range=(min_distance, max_distance))
    ax1.set_title(f'Distance Distribution, {cnt} Centers')
    
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect('equal')
    
    for center in centers:
        circle = plt.Circle(center, radius, fill=False)
        ax2.add_artist(circle)
    ax2.set_title(f'Circle Visualization, var: {curent_var}')
    img_buf = io.BytesIO()
    plt.savefig(img_buf,format='png')
    img = Image.open(img_buf)
    return img

def get_dist_list(check_centers, k = 1):
    global centers
    if k ==1:
        distances, _ = tree.query(check_centers)
    elif k == 2:
        d, _ = tree.query(centers, k=2)
        distances = d[:,1]
    return distances.tolist()

def get_best_radius(center_dists, bins=100):
    min_hist_idxs = []
    hist, bin_edges = np.histogram(center_dists, range=(min_distance, max_distance),bins=bins)
    min_hist = min(hist)
    for i in range(len(hist)):
        if hist[i]==min_hist:
            min_hist_idxs.append(i)
    best_radius = bin_edges[min_hist_idxs] 
    return best_radius

def is_in_region(point):
    global tree, min_distance, max_distance, radius, width, height, centers
    dist, _ = tree.query(point)
    if (min_distance <= dist <= max_distance
        and radius <= point[0] <= width-radius
        and radius <= point[1] <= height-radius):
        return True
    return False

def sample_circle(radius, num_points, center):
    points = []
    c_idxs = []
    for i in range(len(center)):
        for r in radius:
            angles = np.linspace(0, 2*np.pi, num_points)
            x = r * np.cos(angles) + center[i][0]
            y = r * np.sin(angles) + center[i][1]
            p_set = np.column_stack((x, y))
            for p in p_set:
                if is_in_region(p):
                    points.append(p)
                    c_idxs.append(i)
    return points, c_idxs

def get_min_var_idx(points, c_idxs, bins=20):
    global centers
    center_dists = get_dist_list(centers, k=2)
    point_dists = get_dist_list(points)
    hist_var = []
    for i in range(len(point_dists)):
        hist, _ = np.histogram(np.append(center_dists, point_dists[i]), range=(min_distance, max_distance),bins=bins)
        current_var = np.var(hist)
        hist_var.append(current_var)
    min_var = min(hist_var)
    p_idxs = []
    for i in range(len(hist_var)):
        if hist_var[i]==min_var:
            p_idxs.append(i)
    p_idx = np.random.choice(p_idxs)
    c_idx = c_idxs[p_idx]
    return c_idx, p_idx
    

def bfs(centers, cnt=1000, sample_cnt=50):
    global tree, min_distance, max_distance
    queue = centers.copy()
    imgs = []
    while queue:
        samples_radius = get_best_radius(get_dist_list(centers))
        points, c_idxs = sample_circle(samples_radius, sample_cnt, queue)
        if not points:
            break
        c_idx, p_idx = get_min_var_idx(points, c_idxs)
        _ = queue.pop(c_idx)
        p = points[p_idx]
        new_center = p.tolist()
        centers.append(new_center)
        distances = get_dist_list(centers, 2)
        imgs.append(im_show(distances, centers, len(centers), 0))
        tree = KDTree(centers)
        print(f'Center Coordinate {len(centers)}', new_center)
        queue.append(new_center)
        if len(centers)>=cnt:
            return centers, imgs
    return centers, imgs

centers, imgs = bfs(centers, cnt=1000, sample_cnt=50)
distances = get_dist_list(centers, k=2)

result = im_show(distances, centers, len(centers), 0)
result.save("result.png")
imgs[0].save("process.gif",format='GIF',append_images=imgs,save_all=True,duration=100,loop=0)