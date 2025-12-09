import numpy as np
import torch

from ..pose_utils import draw_aapose_by_meta_new


def render_pose_canvases(pose_metas_primary, pose_metas_all, frame_person_indices, width, height,
                         body_stick_width=-1, hand_stick_width=-1, draw_head=True):
    draw_hand = hand_stick_width != 0
    frames_to_draw = len(frame_person_indices) if frame_person_indices is not None else len(pose_metas_primary)
    pose_images = []
    for idx in range(frames_to_draw):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_person_indices is not None:
            indices = frame_person_indices[idx]
            if indices and pose_metas_all:
                metas_to_draw = [pose_metas_all[min(i, len(pose_metas_all) - 1)] for i in indices]
            else:
                metas_to_draw = [pose_metas_primary[min(idx, len(pose_metas_primary) - 1)]]
        else:
            metas_to_draw = [pose_metas_primary[idx]]

        pose_image = canvas
        for meta in metas_to_draw:
            pose_image = draw_aapose_by_meta_new(
                pose_image,
                meta,
                draw_hand=draw_hand,
                draw_head=draw_head,
                body_stick_width=body_stick_width,
                hand_stick_width=hand_stick_width,
            )
        pose_images.append(pose_image)

    pose_images_np = np.stack(pose_images, 0)
    pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0
    return pose_images_tensor
