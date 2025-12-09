import math
import numpy as np
import cv2


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):
    target_ratio = orig_w / max(orig_h, 1e-5)

    def check_valid(w, h):
        if w <= 0 or h <= 0:
            return False
        return (w * h <= target_area and w % divisor == 0 and h % divisor == 0)

    def get_ratio_diff(w, h):
        return abs(w / h - target_ratio)

    def round_to_divisor(value, round_up=False):
        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    possible_sizes = []
    max_area_h = int(np.sqrt(target_area / max(target_ratio, 1e-5)))
    max_area_w = int(max_area_h * target_ratio)

    max_h = round_to_divisor(max_area_h, round_up=True)
    max_w = round_to_divisor(max_area_w, round_up=True)

    for h in range(divisor, max_h + divisor, divisor):
        ideal_w = h * target_ratio
        w_down = round_to_divisor(ideal_w)
        w_up = round_to_divisor(ideal_w, round_up=True)
        for w in (w_down, w_up):
            if check_valid(w, h):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    if not possible_sizes:
        raise ValueError("No suitable resize candidates")

    possible_sizes.sort(key=lambda size: (-size[0] * size[1], size[2]))
    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    ori_height, ori_width = img_ori.shape[:2]
    channel = 1 if img_ori.ndim == 2 else img_ori.shape[2]

    canvas = np.zeros((height, width, channel), dtype=img_ori.dtype)
    for c in range(channel):
        canvas[:, :, c] = padding_color[c % len(padding_color)]

    src_aspect = ori_height / max(ori_width, 1e-5)
    dst_aspect = height / max(width, 1e-5)

    if src_aspect > dst_aspect:
        new_width = int(height / max(src_aspect, 1e-5))
        resized = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        if channel == 1 and resized.ndim == 2:
            resized = resized[:, :, None]
        pad = (width - new_width) // 2
        canvas[:, pad:pad + new_width] = resized
    else:
        new_height = int(width * src_aspect)
        resized = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        if channel == 1 and resized.ndim == 2:
            resized = resized[:, :, None]
        pad = (height - new_height) // 2
        canvas[pad:pad + new_height] = resized

    return canvas


def resize_by_area(image, target_area, keep_aspect_ratio=True, divisor=64, padding_color=(0, 0, 0)):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except ValueError:
        aspect_ratio = w / max(h, 1e-5)
        if keep_aspect_ratio:
            new_h = math.sqrt(target_area / max(aspect_ratio, 1e-5))
            new_w = target_area / max(new_h, 1e-5)
        else:
            new_w = new_h = math.sqrt(target_area)
        new_w = int((new_w // divisor) * divisor)
        new_h = int((new_h // divisor) * divisor)

    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR
    return padding_resize(image, height=new_h, width=new_w, padding_color=padding_color, interpolation=interpolation)


def resize_to_bounds(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR, extra_padding=64, crop_target_image=None):
    def _compute_crop_bounds(source):
        if source.ndim == 2:
            mask = source > 0
        else:
            mask = np.any(source != 0, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return 0, source.shape[0], 0, source.shape[1]
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return y0, y1, x0, x1

    if crop_target_image is not None:
        y0, y1, x0, x1 = _compute_crop_bounds(crop_target_image)
    else:
        y0, y1, x0, x1 = _compute_crop_bounds(img_ori)

    pad_y0 = max(y0 - extra_padding, 0)
    pad_y1 = min(y1 + extra_padding, img_ori.shape[0])
    pad_x0 = max(x0 - extra_padding, 0)
    pad_x1 = min(x1 + extra_padding, img_ori.shape[1])
    crop_img = img_ori[pad_y0:pad_y1, pad_x0:pad_x1]

    ori_height, ori_width = crop_img.shape[:2]
    channel = 1 if crop_img.ndim == 2 else crop_img.shape[2]

    canvas = np.zeros((height, width, channel), dtype=crop_img.dtype)
    for c in range(channel):
        canvas[:, :, c] = padding_color[c % len(padding_color)]

    crop_aspect = ori_width / max(ori_height, 1e-5)
    target_aspect = width / max(height, 1e-5)
    if crop_aspect > target_aspect:
        new_width = width
        new_height = int(width / max(crop_aspect, 1e-5))
    else:
        new_height = height
        new_width = int(height * crop_aspect)

    resized = cv2.resize(crop_img, (new_width, new_height), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    y_pad = (height - new_height) // 2
    x_pad = (width - new_width) // 2
    canvas[y_pad:y_pad + new_height, x_pad:x_pad + new_width, :] = resized
    return canvas
