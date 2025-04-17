from .nodes import LoadKimiVLImage, LoadKimiVLModel, KimiVL, SaveKimiVLText

NODE_CLASS_MAPPINGS = {
    "LoadKimiVLImage": LoadKimiVLImage,
    "LoadKimiVLModel": LoadKimiVLModel,
    "KimiVL": KimiVL,
    "SaveKimiVLText": SaveKimiVLText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadKimiVLImage": "Load Kimi VL Image",
    "LoadKimiVLModel": "Load Kimi VL Model",
    "KimiVL": "Kimi VL",
    "SaveKimiVLText": "Save Kimi VL Text",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
