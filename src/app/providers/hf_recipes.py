"""
Hugging Face recipe generation helper for the recommendation endpoint.
Loads a seq2seq model once and exposes a small generation helper that returns raw text.
Designed to be optional: if HF_MODEL_ID is not set or loading fails, callers should fallback.
"""

from __future__ import annotations

import os
from typing import List, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Cache the pipeline so the model is loaded only once per process.
_HF_PIPE = None


def _device_arg() -> int:
    """
    Transformers pipeline expects an int device:
    -1 for CPU, 0 for first CUDA/MPS device.
    We keep it simple: HF_DEVICE=cuda uses device 0, anything else => CPU.
    """
    device = os.getenv("HF_DEVICE", "cpu").lower()
    if device == "cuda":
        return 0
    return -1


def get_hf_pipeline() -> Optional[object]:
    """
    Lazily initialize and cache a text2text-generation pipeline.
    Returns None if HF_MODEL_ID is not configured or loading fails.
    """
    global _HF_PIPE
    if _HF_PIPE is not None:
        return _HF_PIPE

    model_id = os.getenv("HF_MODEL_ID")
    if not model_id:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _HF_PIPE = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=_device_arg(),
        )
    except Exception:
        _HF_PIPE = None
    return _HF_PIPE


def generate_hf_recipes(prompts: List[str], **generation_kwargs) -> Optional[List[str]]:
    """
    Generate recipe text for a batch of prompts. Each prompt should already include any prefix.
    Returns None on any loading or inference failure.
    """
    pipe = get_hf_pipeline()
    if pipe is None:
        return None

    try:
        outputs = pipe(prompts, **generation_kwargs)
        # Pipeline returns list[dict], each with a "generated_text" key.
        recipes = [o.get("generated_text", "").strip() for o in outputs if isinstance(o, dict)]
        return [r for r in recipes if r]
    except Exception:
        return None
