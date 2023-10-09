import json
import math
import os

import cv2
import numpy as np
from PIL import Image
from numpy.random import multivariate_normal
from tqdm import tqdm


def gen_gaussian2d(shape, sigma=1):
    h, w = [_ // 2 for _ in shape]
    y, x = np.ogrid[-h : h + 1, -w : w + 1]

    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(density, center, radius, k=1, delte=6, overlap="add"):
    # diameter = 2 * radius + 1
    radius_2 = [radius[0] * 2 + 1, radius[1] * 2 + 1]
    gaussian = gen_gaussian2d(radius_2, sigma=min(radius_2[0], radius_2[1]) / delte)
    x, y = int(center[0]), int(center[1])
    height, width = density.shape[0:2]
    left, right = min(x, radius[1]), min(width - x, radius[1] + 1)
    top, bottom = min(y, radius[0]), min(height - y, radius[0] + 1)
    if overlap == "max":
        masked_density = density[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius[0] - top : radius[0] + bottom, radius[1] - left : radius[1] + right
        ]
        np.maximum(masked_density, masked_gaussian * k, out=masked_density)
    elif overlap == "add":
        density[y - top : y + bottom, x - left : x + right] += gaussian[
            radius[0] - top : radius[0] + bottom, radius[1] - left : radius[1] + right
        ]
    else:
        raise NotImplementedError


def _min_dis_global(point, points, win_h, win_w):
    """
    points: m x 2, m x [x, y]
    """
    p_x, p_y = point
    for o_point in points:
        dis_y, dis_x = abs(p_y - o_point[1]), abs(p_x - o_point[0])
        if dis_y < 1 and dis_x < 1:
            continue
        if dis_y > win_h * 2 or dis_x > win_w * 2:
            continue
        else:
            if dis_y / win_h > dis_x / win_w:
                sca = dis_y / 2 / win_h
            else:
                sca = dis_x / 2 / win_w
            win_h, win_w = win_h * sca, win_w * sca
    win_h, win_w = max(int(win_h), 1), max(int(win_w), 1)
    return win_h, win_w


def points2density(points, radius_backup=None):
    """
    points: m x 2, m x [x, y]
    """
    num_points = points.shape[0]
    density = np.zeros(image_size, dtype=np.float32)  # [h, w]
    if num_points == 0:
        return np.zeros(image_size, dtype=np.float32)
    if num_points == 1:
        draw_gaussian(density, points[0], radius_backup, overlap="max")
    else:
        for point in points:
            dis_h, dis_w = _min_dis_global(point, points, radius_backup[0], radius_backup[1])
            draw_gaussian(density, point, [dis_h, dis_w], overlap="max")
    return density


if __name__ == "__main__":
    gt_dir = "./FSC147_384_V2/gt_density_map/"
    anno_file = './annotation_FSC147_384.json'
    data_split_file = './Train_Test_Val_FSC_147.json'
    im_dir = 'FSC147_384_V2/images_384_VarV2'
    os.makedirs(gt_dir, exist_ok=True)

    with open(anno_file) as f:  # 标签文件
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    im_ids = data_split['train'] + data_split['val'] + data_split['test']
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        boxers = anno['box_examples_coordinates']  # 示例框坐标
        dots = np.array(anno['points'])  # 目标物体点坐标
        cnt_gt = len(dots)
        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        w_h = image.size
        image_size = [w_h[1], w_h[0]]

        rects = list()
        a_w = a_h = 0.0
        for box in boxers:  # 取两个点即可
            xl = box[0][0]
            yl = box[0][1]
            xr = box[2][0]
            yr = box[2][1]

            a_w += (xr - xl) / 2
            a_h += (yr - yl) / 2
        a_w = max(int(a_w / len(boxers)), 1)
        a_h = max(int(a_h / len(boxers)), 1)

        density = points2density(dots, [a_h, a_w])

        if not cnt_gt == 0:
            cnt_cur = density.sum()
            density = density / cnt_cur * cnt_gt

        filename_ = os.path.splitext(im_id)[0]
        save_path = os.path.join(gt_dir, filename_ + ".npy")
        np.save(save_path, density)
        print(f"Success: generate gt density map for {im_id}")
