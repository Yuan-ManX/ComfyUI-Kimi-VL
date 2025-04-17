from PIL import Image
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class LoadImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "./figures/demo.png"
                })
            }
        }

    RETURN_TYPES = ("PIL_IMAGE", "STRING",)
    RETURN_NAMES = ("image", "image_path",)
    FUNCTION = "load_image"
    CATEGORY = "Kimi-VL"

    def load_image(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = Image.open(image_path)
        return (image, image_path)


class LoadKimiVLModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "moonshotai/Kimi-VL-A3B-Instruct"
                }),
                "use_flash_attention": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("KIMI_MODEL", "KIMI_PROCESSOR",)
    RETURN_NAMES = ("model", "processor",)
    FUNCTION = "load_model"
    CATEGORY = "Kimi-VL"

    def load_model(self, model_path, use_flash_attention):
        if use_flash_attention == "enable":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return (model, processor)


class KimiVL:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KIMI_MODEL",),
                "processor": ("KIMI_PROCESSOR",),
                "image": ("PIL_IMAGE",),
                "image_path": ("STRING",),
                "prompt": ("STRING", {
                    "default": "What is the dome building in the picture? Think step by step."
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048,
                    "step": 1,
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run_inference"
    CATEGORY = "Kimi-VL"

    def run_inference(self, model, processor, image, image_path, prompt, max_new_tokens):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return (response,)


class SaveText:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "response": ("STRING",),
                "save_path": ("STRING", {
                    "default": "./output_kimi.txt"
                }),
                "print_to_console": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_response",)
    FUNCTION = "output"
    CATEGORY = "Kimi-VL"

    def output(self, response, save_path, print_to_console):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response)

        if print_to_console == "enable":
            print("[Kimi-VL] Model output results：\n", response)
            print(f"[Kimi-VL] Saved to: {save_path}")

        return (response,)



