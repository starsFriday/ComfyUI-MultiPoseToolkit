import json
import numpy as np
import torch
import cv2
from tqdm import tqdm

from comfy.utils import ProgressBar

from ..data import AAPoseMeta, load_pose_metas_from_kp2ds_seq, bbox_from_detector, crop
from ..utils import render_pose_canvases


def _sanitize_bbox(raw_bbox, width, height):
    bbox = np.zeros(4, dtype=np.float32)
    raw = np.array(raw_bbox, dtype=np.float32).flatten()
    bbox[:min(len(raw), 4)] = raw[:4]
    width = max(int(width), 1)
    height = max(int(height), 1)
    bbox[0] = np.clip(bbox[0], 0, width - 1)
    bbox[1] = np.clip(bbox[1], 0, height - 1)
    bbox[2] = np.clip(bbox[2], bbox[0] + 1, width)
    bbox[3] = np.clip(bbox[3], bbox[1] + 1, height)
    if bbox[2] <= bbox[0]:
        bbox[2] = min(width, bbox[0] + 1)
    if bbox[3] <= bbox[1]:
        bbox[3] = min(height, bbox[1] + 1)
    return bbox


def _bbox_to_int_bounds(bbox, width, height):
    width = max(int(width), 1)
    height = max(int(height), 1)
    x0 = int(np.clip(np.floor(bbox[0]), 0, width - 1))
    y0 = int(np.clip(np.floor(bbox[1]), 0, height - 1))
    x1 = int(np.clip(np.ceil(bbox[2]) - 1, x0, width - 1))
    y1 = int(np.clip(np.ceil(bbox[3]) - 1, y0, height - 1))
    return x0, y0, x1, y1


def _sample_points_inside_bbox(rng, bounds, count):
    if count <= 0:
        return []
    x0, y0, x1, y1 = bounds
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    aspect = width / max(height, 1)
    cols = max(1, int(np.ceil(np.sqrt(count * max(aspect, 0.25)))))
    rows = max(1, int(np.ceil(count / cols)))
    xs = np.linspace(x0, x1, num=cols)
    ys = np.linspace(y0, y1, num=rows)
    grid = []
    jitter_x = max(1.0, width / cols) * 0.5
    jitter_y = max(1.0, height / rows) * 0.5
    for y in ys:
        for x in xs:
            px = x + rng.uniform(-jitter_x, jitter_x)
            py = y + rng.uniform(-jitter_y, jitter_y)
            px = int(np.clip(round(px), x0, x1))
            py = int(np.clip(round(py), y0, y1))
            grid.append({"x": px, "y": py})
    order = rng.permutation(len(grid))
    return [grid[i] for i in order[:count]]


def _point_inside_bounds(x, y, bounds):
    x0, y0, x1, y1 = bounds
    return x0 <= x <= x1 and y0 <= y <= y1


def _sample_points_outside_bboxes(rng, width, height, forbidden_bounds, count):
    if count <= 0:
        return []
    width = max(int(width), 1)
    height = max(int(height), 1)
    points = []
    attempts = 0
    max_attempts = max(100, count * 25)
    while len(points) < count and attempts < max_attempts:
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        if not any(_point_inside_bounds(x, y, b) for b in forbidden_bounds):
            points.append({"x": x, "y": y})
        attempts += 1
    if len(points) < count:
        # Fall back to deterministic padding near image edges
        remaining = count - len(points)
        for idx in range(remaining):
            edge_x = 0 if idx % 2 == 0 else width - 1
            edge_y = (idx * height) // max(remaining, 1)
            edge_y = int(np.clip(edge_y, 0, height - 1))
            if not any(_point_inside_bounds(edge_x, edge_y, b) for b in forbidden_bounds):
                points.append({"x": edge_x, "y": edge_y})
            else:
                points.append({"x": edge_x, "y": int(np.clip(edge_y + 1, 0, height - 1))})
    return points


def _clamp_point_to_bounds(x, y, bounds):
    x0, y0, x1, y1 = bounds
    px = int(np.clip(round(float(x)), x0, x1))
    py = int(np.clip(round(float(y)), y0, y1))
    return px, py


def _estimate_min_distance(bounds, count):
    if count <= 1:
        return 1.0
    width = max(1, bounds[2] - bounds[0] + 1)
    height = max(1, bounds[3] - bounds[1] + 1)
    cell = np.sqrt((width * height) / float(max(count, 1)))
    return max(2.0, cell * 0.45)


def _select_spread_points(candidates, count, rng, min_distance):
    if count <= 0:
        return []
    if not candidates:
        return []
    min_distance_sq = float(min_distance) * float(min_distance)
    order = rng.permutation(len(candidates))
    shuffled = [candidates[i] for i in order]
    selected = []
    for pt in shuffled:
        if all(((pt["x"] - prev["x"]) ** 2 + (pt["y"] - prev["y"]) ** 2) >= min_distance_sq for prev in selected):
            selected.append(pt)
            if len(selected) == count:
                return selected
    if len(selected) < count and min_distance > 2.0:
        return _select_spread_points(candidates, count, rng, min_distance * 0.75)
    while len(selected) < count:
        selected.append(shuffled[len(selected) % len(shuffled)])
    return selected[:count]


def _generate_candidates_from_keypoints(coords, bounds, rng, jitter):
    candidates = []
    for coord in coords:
        x, y = coord
        if jitter > 0:
            x = float(x) + rng.normal(0, jitter)
            y = float(y) + rng.normal(0, jitter)
        px, py = _clamp_point_to_bounds(x, y, bounds)
        candidates.append({"x": px, "y": py})
    return candidates


def _generate_interpolated_candidates(coords, bounds, rng, jitter, needed):
    coords = np.asarray(coords, dtype=np.float32)
    total = coords.shape[0]
    if total < 2 or needed <= 0:
        return []
    candidates = []
    for _ in range(needed):
        idx_a, idx_b = rng.choice(total, size=2, replace=True)
        a = coords[idx_a]
        b = coords[idx_b]
        t = rng.random()
        point = a * t + b * (1.0 - t)
        if jitter > 0:
            point = point + rng.normal(0, jitter, size=2)
        px, py = _clamp_point_to_bounds(point[0], point[1], bounds)
        candidates.append({"x": px, "y": py})
    return candidates


def _sample_positive_points(
    rng,
    bbox_bounds,
    count,
    keypoints=None,
    conf_threshold=0.3,
    jitter=5.0,
):
    if count <= 0:
        return []

    jitter = max(float(jitter), 0.0)

    candidate_points = []
    keypoints_array = None
    if keypoints is not None:
        keypoints_array = np.asarray(keypoints, dtype=np.float32)
        if keypoints_array.ndim == 3:
            keypoints_array = keypoints_array[0]
        if keypoints_array.shape[-1] < 2:
            keypoints_array = None

    if keypoints_array is not None:
        if keypoints_array.shape[-1] == 2:
            confidences = np.ones((keypoints_array.shape[0],), dtype=np.float32)
        else:
            confidences = keypoints_array[:, 2]
        valid_mask = confidences >= float(conf_threshold)
        valid_coords = keypoints_array[valid_mask][:, :2]
        valid_coords = valid_coords[np.all(np.isfinite(valid_coords), axis=1)]
        if valid_coords.size:
            valid_coords[:, 0] = np.clip(valid_coords[:, 0], bbox_bounds[0], bbox_bounds[2])
            valid_coords[:, 1] = np.clip(valid_coords[:, 1], bbox_bounds[1], bbox_bounds[3])
            coords_int = np.round(valid_coords).astype(int)
            if coords_int.size:
                _, unique_idx = np.unique(coords_int, axis=0, return_index=True)
                coords_unique = valid_coords[unique_idx]
            else:
                coords_unique = valid_coords
            if coords_unique.size:
                order = rng.permutation(len(coords_unique))
                coords_unique = coords_unique[order]
                candidate_points.extend(
                    _generate_candidates_from_keypoints(coords_unique, bbox_bounds, rng, jitter)
                )
                interp_needed = max(count * 4, len(coords_unique) * 2)
                candidate_points.extend(
                    _generate_interpolated_candidates(coords_unique, bbox_bounds, rng, jitter, interp_needed)
                )

    if not candidate_points:
        candidate_points = _sample_points_inside_bbox(rng, bbox_bounds, max(count * 2, count))
    else:
        candidate_points.extend(_sample_points_inside_bbox(rng, bbox_bounds, max(count, 4)))

    min_distance = _estimate_min_distance(bbox_bounds, count)
    return _select_spread_points(candidate_points, count, rng, min_distance)


def _format_points_output(entries):
    if not entries:
        return "[]"
    if len(entries) == 1:
        return json.dumps(entries[0]["points"], ensure_ascii=False)
    return json.dumps(entries, ensure_ascii=False)


def _format_bbox_output(entries):
    if not entries:
        return []
    return [entry["bbox"] for entry in entries]


class MultiPersonPoseExtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "process"
    CATEGORY = "WanMultiPose"

    def process(self, model, images):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        batch, H, W, _ = images.shape
        np_images = images.numpy()
        shape = np.array([[H, W]], dtype=np.float32)

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        detect_multi_persons = True
        frame_person_bboxes = []
        for img in tqdm(np_images, total=len(np_images), desc="Detecting bboxes (multi-person)"):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape,
                single_person=not detect_multi_persons,
            )[0]
            person_boxes = []
            for det in detections:
                bbox = np.array(det["bbox"], dtype=np.float32)
                if bbox.shape[0] < 4:
                    continue
                bbox[0] = np.clip(bbox[0], 0, W - 1)
                bbox[1] = np.clip(bbox[1], 0, H - 1)
                bbox[2] = np.clip(bbox[2], bbox[0] + 1, W)
                bbox[3] = np.clip(bbox[3], bbox[1] + 1, H)
                person_boxes.append(bbox)
            if not person_boxes:
                person_boxes = [np.array([0, 0, W, H, 1.0], dtype=np.float32)]
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            frame_person_bboxes.append(person_boxes)

        detector.cleanup()

        total_pose_ops = sum(max(1, len(p)) for p in frame_person_bboxes)
        comfy_pbar = ProgressBar(len(np_images) + total_pose_ops)
        comfy_pbar.update_absolute(len(np_images))

        kp2ds_primary = []
        kp2ds_all = []
        frame_person_indices = [[] for _ in range(batch)]

        for frame_idx, (img, bbox_list) in enumerate(tqdm(zip(np_images, frame_person_bboxes), total=len(np_images), desc="Extracting keypoints")):
            if not bbox_list:
                bbox_list = [np.array([0, 0, W, H, 1.0], dtype=np.float32)]
            for person_idx, bbox in enumerate(bbox_list):
                bbox_xyxy = np.array(bbox[:4], dtype=np.float32)
                if (bbox_xyxy[2] - bbox_xyxy[0]) < 10 or (bbox_xyxy[3] - bbox_xyxy[1]) < 10:
                    bbox_xyxy = np.array([0, 0, W, H], dtype=np.float32)
                center, scale = bbox_from_detector(bbox_xyxy, input_resolution, rescale=rescale)
                cropped = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
                img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
                img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
                keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
                kp2ds_all.append(keypoints)
                frame_person_indices[frame_idx].append(len(kp2ds_all) - 1)
                if person_idx == 0:
                    kp2ds_primary.append(keypoints)

        pose_model.cleanup()

        kp2ds_primary = np.concatenate(kp2ds_primary, 0)
        kp2ds_all = np.concatenate(kp2ds_all, 0)

        pose_metas_primary = load_pose_metas_from_kp2ds_seq(kp2ds_primary, width=W, height=H)
        pose_metas_all = load_pose_metas_from_kp2ds_seq(kp2ds_all, width=W, height=H)

        pose_metas_primary_aapo = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas_primary]
        pose_metas_all_aapo = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas_all]

        pose_images_tensor = render_pose_canvases(
            pose_metas_primary=pose_metas_primary_aapo,
            pose_metas_all=pose_metas_all_aapo,
            frame_person_indices=frame_person_indices,
            width=W,
            height=H,
        )

        return (pose_images_tensor,)


class MultiPoseCoordinateSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "positive_points": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "negative_points": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "person_index": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
                "detect_multi_persons": ("BOOLEAN", {"default": True}),
                "positive_mode": (["pose", "bbox"],{"default": "pose"}),
                "pose_conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_jitter": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX")
    RETURN_NAMES = ("positive_coords", "negative_coords", "bbox")
    FUNCTION = "sample"
    CATEGORY = "WanMultiPose"

    def sample(
        self,
        model,
        images,
        positive_points=3,
        negative_points=5,
        person_index=0,
        seed=0,
        detect_multi_persons=True,
        positive_mode="pose",
        pose_conf_threshold=0.3,
        pose_jitter=5.0,
    ):
        detector = model.get("yolo")
        if detector is None:
            raise ValueError("model must contain a 'yolo' detector.")

        pose_model = model.get("vitpose")
        positive_mode = (positive_mode or "pose").lower()
        use_pose_sampling = positive_mode == "pose"
        if use_pose_sampling and pose_model is None:
            raise ValueError("model must contain a 'vitpose' runtime when positive_mode='pose'.")
        pose_conf_threshold = float(pose_conf_threshold)
        pose_jitter = float(pose_jitter)

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            np_images = images.numpy()
        elif isinstance(images, np.ndarray):
            np_images = images
        else:
            np_images = np.array(images)
        batch, H, W, _ = np_images.shape
        shape = np.array([[H, W]], dtype=np.float32)

        detector.reinit()
        frame_person_bboxes = []
        for img in tqdm(np_images, total=len(np_images), desc="Detecting bboxes (coords)"):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape,
                single_person=not detect_multi_persons,
            )[0]
            person_boxes = []
            for det in detections:
                bbox = _sanitize_bbox(det["bbox"], W, H)
                person_boxes.append(bbox)
            if not person_boxes:
                person_boxes = [_sanitize_bbox([0, 0, W, H], W, H)]
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            frame_person_bboxes.append(person_boxes)
        detector.cleanup()

        frame_person_keypoints = [[] for _ in frame_person_bboxes]
        if use_pose_sampling:
            pose_model.reinit()
            IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
            IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
            input_resolution = (256, 192)
            rescale = 1.25
            for frame_idx, (img, boxes) in enumerate(
                tqdm(
                    zip(np_images, frame_person_bboxes),
                    total=len(np_images),
                    desc="Refining poses (coords)",
                )
            ):
                kp_list = []
                if not boxes:
                    boxes = [_sanitize_bbox([0, 0, W, H], W, H)]
                for bbox in boxes:
                    bbox_xyxy = np.array(bbox[:4], dtype=np.float32)
                    if (bbox_xyxy[2] - bbox_xyxy[0]) < 10 or (bbox_xyxy[3] - bbox_xyxy[1]) < 10:
                        bbox_xyxy = np.array([0, 0, W, H], dtype=np.float32)
                    center, scale = bbox_from_detector(bbox_xyxy, input_resolution, rescale=rescale)
                    cropped = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
                    img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
                    img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
                    keypoints = pose_model(
                        img_norm[None],
                        np.array(center, dtype=np.float32)[None],
                        np.array(scale, dtype=np.float32)[None],
                    )
                    if isinstance(keypoints, torch.Tensor):
                        keypoints = keypoints.detach().cpu().numpy()
                    keypoints = np.asarray(keypoints, dtype=np.float32)
                    if keypoints.ndim >= 3:
                        keypoints = keypoints[0]
                    kp_list.append(keypoints)
                frame_person_keypoints[frame_idx] = kp_list
            pose_model.cleanup()

        rng = np.random.default_rng(int(seed))
        positive_entries = []
        negative_entries = []
        bbox_entries = []

        for frame_idx, boxes in enumerate(frame_person_bboxes):
            if not boxes:
                boxes = [_sanitize_bbox([0, 0, W, H], W, H)]
            target_idx = min(max(int(person_index), 0), len(boxes) - 1)
            target_bbox = boxes[target_idx]
            bbox_bounds = _bbox_to_int_bounds(target_bbox, W, H)
            forbidden_bounds = [_bbox_to_int_bounds(b, W, H) for b in boxes]
            target_keypoints = None
            if use_pose_sampling:
                kp_list = frame_person_keypoints[frame_idx] if frame_idx < len(frame_person_keypoints) else []
                if kp_list and target_idx < len(kp_list):
                    target_keypoints = kp_list[target_idx]
            pos_points = _sample_positive_points(
                rng,
                bbox_bounds,
                positive_points,
                keypoints=target_keypoints,
                conf_threshold=pose_conf_threshold,
                jitter=pose_jitter,
            )
            neg_points = _sample_points_outside_bboxes(rng, W, H, forbidden_bounds, negative_points)
            positive_entries.append({"image_index": frame_idx, "points": pos_points})
            negative_entries.append({"image_index": frame_idx, "points": neg_points})
            bbox_entries.append({"image_index": frame_idx, "bbox": tuple(int(v) for v in bbox_bounds)})

        positive_output = _format_points_output(positive_entries)
        negative_output = _format_points_output(negative_entries)
        bbox_output = _format_bbox_output(bbox_entries)

        return (positive_output, negative_output, [bbox_output])
