"""
Lightweight vision caption helper with graceful fallback.
Attempts BLIP captioning if transformers/PIL are available; otherwise returns a stub description.
Kept small so CPU-only hosts don't crash—use a GPU host for production quality.
"""

from __future__ import annotations

from typing import Tuple
import io


def describe_image_from_bytes(data: bytes) -> Tuple[str, str]:
    """
    Return (description, debug_source). On failure, returns a generic description.
    """
    try:
        from PIL import Image  # type: ignore
        import torch  # type: ignore
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = processor(image, return_tensors="pt", padding=True).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            length_penalty=0.6,
            no_repeat_ngram_size=2,
        )
        desc = processor.decode(out[0], skip_special_tokens=True)
        return desc, f"blip:{device}"
    except Exception:
        return "A photo of a plated dish (fallback description).", "fallback"
