import numpy as np
import torch
import cv2
from tqdm import tqdm

from comfy.utils import ProgressBar

from ..data import AAPoseMeta, load_pose_metas_from_kp2ds_seq, bbox_from_detector, crop
from ..utils import render_pose_canvases


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
