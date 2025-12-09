import numpy as np

def get_face_bboxes(keypoints_face, scale, image_shape):
    """Return expanded facial bounding box (x0, x1, y0, y1)."""
    h, w = image_shape
    kp2ds_face = keypoints_face.copy() * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y
    initial_area = max(initial_width * initial_height, 1.0)

    expanded_area = initial_area * scale
    new_width = np.sqrt(expanded_area * (initial_width / max(initial_height, 1e-5)))
    new_height = np.sqrt(expanded_area * (initial_height / max(initial_width, 1e-5)))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]
