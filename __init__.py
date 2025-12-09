def _lazy_import():
    from .toolkit.models.runtime import OnnxRuntimeLoader
    from .toolkit.pipeline.detector import MultiPersonPoseExtraction
    return {
        "OnnxRuntimeLoader": OnnxRuntimeLoader,
        "MultiPersonPoseExtraction": MultiPersonPoseExtraction,
    }

_NODE_CACHE = None

def _node_map():
    global _NODE_CACHE
    if _NODE_CACHE is None:
        _NODE_CACHE = _lazy_import()
    return _NODE_CACHE

NODE_CLASS_MAPPINGS = _node_map()
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxRuntimeLoader": "MultiPose ▸ ONNX Loader",
    "MultiPersonPoseExtraction": "MultiPose ▸ Pose Extraction",
}
