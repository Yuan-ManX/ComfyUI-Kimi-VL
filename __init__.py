from .nodes import LoadImage, LoadKimiVLModel, KimiVL, SaveText

NODE_CLASS_MAPPINGS = {
    "LoadImage": LoadImage,
    "LoadKimiVLModel": LoadKimiVLModel,
    "KimiVL": KimiVL,
    "SaveText": SaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImage": "Load Image",
    "LoadKimiVLModel": "Load Kimi VL Model",
    "KimiVL": "Kimi VL",
    "SaveText": "Save Text",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
