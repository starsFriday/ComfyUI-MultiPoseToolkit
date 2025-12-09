import os
import numpy as np
import torch
import onnxruntime
import folder_paths
import cv2

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))


def _provider(device):
    if device == "CUDAExecutionProvider":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [device]


class SimpleOnnxRuntime:
    def __init__(self, checkpoint, device="CUDAExecutionProvider", **kwargs):
        self.checkpoint = checkpoint
        self.init_kwargs = kwargs
        self.provider = _provider(device)
        self.session = onnxruntime.InferenceSession(checkpoint, providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = np.array(self.session.get_inputs()[0].shape[2:])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cleanup(self):
        if self.session is not None:
            del self.session
            self.session = None

    def reinit(self, provider=None):
        if provider is not None:
            self.provider = provider
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.checkpoint, providers=self.provider)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_resolution = np.array(self.session.get_inputs()[0].shape[2:])


class YoloRuntime(SimpleOnnxRuntime):
    def __init__(self, checkpoint, device="CUDAExecutionProvider", threshold_conf=0.05,
                 threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5,
                 threshold_bbox_shape_ratio=0.4, cat_id=(1,), select_type='max', strict=True, sorted_func=None):
        super().__init__(checkpoint, device=device)
        self.input_width = input_resolution[0]
        self.input_height = input_resolution[1]
        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func

    def postprocess(self, output, shape_raw, cat_id=(1,)):
        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)

        rows = outputs.shape[0]
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]
            mask = (max_scores >= self.threshold_conf) & (classid != 3.14159)
            max_scores = max_scores[mask]
            classid = classid[mask]
            boxes = outputs[:, :4][mask]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)
        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            mask = max_scores >= self.threshold_conf
            classid = np.argmax(classes_scores[mask], -1)
            valid_mask = classid != 3.14159
            classes_scores = classes_scores[mask][valid_mask]
            max_scores = max_scores[mask][valid_mask]
            classid = classid[valid_mask]
            xywh = outputs[:, :4][mask][valid_mask]
            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]
            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)

        results = []
        for i in indices:
            box = boxes[i]
            x0, y0, w, h = box
            results.append([x0, y0, x0 + w, y0 + h, scores[i], class_ids[i]])
        return np.array(results)

    def process_results(self, results, shape_raw, cat_id=(1,), single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results
        person_results = []
        person_count = 0
        if len(det_results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1
            bboxes = []
            idx_list = []
            for i in range(det_results.shape[0]):
                bbox = det_results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape
            det_results = det_results[idx_list]
            for i in range(det_results.shape[0]):
                bbox = det_results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i
            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
            if max_idx != -1:
                person_count = 1
                person_results.append({'bbox': det_results[max_idx, :5], 'track_id': int(0)})
            for i in range(det_results.shape[0]):
                bbox = det_results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person_results.append({'bbox': det_results[i, :5], 'track_id': int(person_count - 1)})
            return person_results
        return None

    def forward(self, img, shape_raw, **kwargs):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: img})[0]
        person_results = [[{'bbox': np.array([0., 0., 1. * shape_raw[i][1], 1. * shape_raw[i][0], -1]), 'track_id': -1}]
                          for i in range(len(outputs))]
        for i in range(len(outputs)):
            result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
            result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, **kwargs)
            if result is not None and len(result) != 0:
                person_results[i] = result
        return person_results


class ViTPoseRuntime(SimpleOnnxRuntime):
    def __init__(self, checkpoint, device="CUDAExecutionProvider", **kwargs):
        super().__init__(checkpoint, device=device)

    def forward(self, img, center, scale, **kwargs):
        heatmaps = self.session.run([], {self.input_name: img})[0]
        from ..pose_utils.pose2d_utils import keypoints_from_heatmaps
        from ..data.keypoints import keypoints_from_heatmaps
        points = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=center,
            scale=scale * 200,
            unbiased=True,
            use_udp=False)
        return points


class OnnxRuntimeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"),
                                   {"tooltip": "ViTPose ONNX checkpoint"}),
                "yolo_model": (folder_paths.get_filename_list("detection"),
                                {"tooltip": "YOLO ONNX checkpoint"}),
                "onnx_device": (("CUDAExecutionProvider", "CPUExecutionProvider"),
                                 {"default": "CUDAExecutionProvider"}),
            }
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("pose_model",)
    FUNCTION = "load_models"
    CATEGORY = "WanMultiPose"

    def load_models(self, vitpose_model, yolo_model, onnx_device):
        vitpose_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_path = folder_paths.get_full_path_or_raise("detection", yolo_model)
        vitpose = ViTPoseRuntime(vitpose_path, device=onnx_device)
        yolo = YoloRuntime(yolo_path, device=onnx_device)
        return ({"vitpose": vitpose, "yolo": yolo},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return folder_paths.folder_paths_changed()
