import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import collections

# 初始化参数
width = 0.25  
height = 0.25  
radius = 5e-3  
min_distance = 12e-3  
max_distance = 18e-3  

# 初始化列表
# centers = [(5e-3, 5e-3)]  # 圆心坐标列表
# centers = [(width/2, height/2), (width/2-2*radius, height/2-2*radius), (width/2+2*radius, height/2+2*radius)]
# centers = [(width/2, height/2), (width/2-max_distance, height/2-max_distance)]
centers = [
            # (radius, radius), 
            # (width-radius, height-radius),
            # (radius, height-radius),
            # (width-radius, radius),
            (width/2, height/2),
            (width*3/4, height/2),
            (width/2, height/4),
            (width/4, height/4),
            (width/2, height*3/4)]
tree = KDTree(centers)


def im_show(distances, centers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # 绘制距离的分布图（分布直方图）
    ax1.hist(distances, bins=100)
    ax1.set_title('Distance Distribution')
    # 绘制所有的圆（可视化）
    
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect('equal')
    
    for center in centers:
        circle = plt.Circle(center, radius, fill=False)
        ax2.add_artist(circle)
    ax2.set_title('Circle Visualization')
    plt.show()


def sample_annulus(inner_radius, outer_radius, num_samples, center_x, center_y):
    global tree 
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, num_samples))
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    # 转换为笛卡尔坐标
    x = r * np.cos(theta) + center_x
    y = r * np.sin(theta) + center_y
    points = []
    for i in range(len(x)):
        new_center = (x[i], y[i])
        dist, _ = tree.query(new_center)
        if (min_distance <= dist <= max_distance 
            and radius <= x[i] <= width-radius 
            and radius <= y[i] <= height-radius):
            points.append(new_center)
    return np.array(points)

def get_dist_list(centers):
    distances = []
    for i in range(len(centers)):
        dist, _ = tree.query(centers[i], k=2)
        distances.append(dist[1])
    return distances

# def get_point_hist_var(center_distance, points_distance):
#     all_distance = center_distance + points_distance
#     all_distance = np.array(all_distance)
#     # 计算平均值
#     mean = np.mean(all_distance)
#     # 计算每个数据点与平均值之差的绝对值
#     diff = np.abs(points_distance - mean)
#     # 计算总离差和
#     total_diff = np.sum(diff)
#     # 计算每个数据点的贡献率
#     scores = diff/total_diff
#     return scores.tolist()

def get_point_hist_var(center_distance, points_distance):
    hist_vars = []
    for p in points_distance:
        # 计算直方图
        hist, bin_edges = np.histogram(center_distance+p, bins=80)
        # 计算直方图的方差
        hist_variance = np.var(hist)
        hist_vars.append(hist_variance)
    return hist_vars

def bfs(centers, cnt=300, sample_cnt=200):
    global tree, min_distance, max_distance
    queue = collections.deque()
    # visited = set()
    for i in range(len(centers)):
        queue.append(centers[i])
        # visited.add(centers[i])
    while queue:
        # distances = get_dist_list(centers)
        # im_show(distances, centers)
        ver = queue.popleft()
        # idxs = np.random.randint(0, len(centers), 4)
        # centers_choices = [centers[i] for i in idxs]
        # visited.add(c for c in choices)
        # visited.add(ver)
        # for c in centers_choices:
        points = sample_annulus(min_distance, max_distance, sample_cnt, ver[0], ver[1])
        centers_dist = get_dist_list(centers)
        points_dist = get_dist_list(points)
        hist_vars = get_point_hist_var(centers_dist, points_dist)
        # if not points_scores.tolist():
        if not hist_vars:
            continue
        max_hist_var = max(hist_vars)
        # max_hist_var_idx = hist_vars.index(max_hist_var)

        for idx in range(0, len(hist_vars)):
            if hist_vars[idx]>=max_hist_var*0.98:
                new_center = points[idx].tolist()
                dist, _ = tree.query(new_center)
                if (min_distance <= dist <= max_distance):
                    # new_center = points[min_score_idx].tolist()
                    centers.append(new_center)
                    tree = KDTree(centers)
                    print(f'Center Coordinate {len(centers)}', new_center)
                    queue.append(new_center)
                    if len(centers)>=cnt:
                        return centers
                    # visited.add(new_center)                    

    return centers
        
centers = bfs(centers, cnt=1000, sample_cnt=500)
distances = get_dist_list(centers)

im_show(distances, centers)