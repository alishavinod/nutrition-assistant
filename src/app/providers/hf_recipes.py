"""
Hugging Face generation helper for the recommendation endpoint.
Targets instruction/chat models (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct) via text-generation pipeline.
Designed to be optional: if HF_MODEL_ID is not set or loading fails, callers should fallback.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

# Cache pipelines keyed by model_id to avoid reloading on each request.
_HF_CACHE: Dict[str, Tuple[object, str, object, int]] = {}
_HF_LAST_ERROR: Optional[str] = None


def _hf_token() -> Optional[str]:
    """
    Return an access token for gated models if provided via env.
    Checks the common vars used by huggingface_hub/transformers.
    """
    return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")


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


def _param_count(model) -> int:
    """
    Compute parameter count once per model; returns int.
    """
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def get_hf_pipeline(model_id_override: Optional[str] = None):
    """
    Lazily initialize and cache a generation pipeline.
    - Defaults to text-generation (causal LM) for instruction-tuned chat models.
    - Can be overridden with HF_TASK=text2text-generation for seq2seq models.
    Returns (pipe, task, tokenizer, param_count) or (None, None, None, 0) on failure.
    """
    global _HF_LAST_ERROR
    model_id = model_id_override or os.getenv("HF_MODEL_ID")
    if not model_id:
        _HF_LAST_ERROR = "HF_MODEL_ID not set"
        return None, None, None, 0

    if model_id in _HF_CACHE:
        pipe, task, tokenizer, params = _HF_CACHE[model_id]
        # If a previous load failed (pipe is None), try again in case env/token changed.
        if pipe is not None:
            return pipe, task, tokenizer, params
        else:
            _HF_CACHE.pop(model_id, None)

    task = os.getenv("HF_TASK", "text-generation").strip()
    token = _hf_token()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=token)
        if task == "text2text-generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_auth_token=token)
        else:
            # Default to causal LM for chat/instruction models like Llama 3.1.
            model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token)
            task = "text-generation"

        pipe = pipeline(task, model=model, tokenizer=tokenizer, device=_device_arg())
        params = _param_count(model)
        _HF_CACHE[model_id] = (pipe, task, tokenizer, params)
        _HF_LAST_ERROR = None
    except Exception as exc:
        missing_token_hint = ""
        if "gated repo" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            missing_token_hint = (
                " (set HUGGINGFACE_HUB_TOKEN or HF_HUB_TOKEN if the model is gated)"
            )
        _HF_LAST_ERROR = f"load failed for {model_id}: {exc}{missing_token_hint}"
        return None, None, None, 0
    return _HF_CACHE[model_id]


def generate_hf_recipes(prompts: List[str], model_id: Optional[str] = None, **generation_kwargs) -> Optional[List[str]]:
    """
    Generate recipe text for a batch of prompts. Each prompt should already include any prefix.
    Returns None on any loading or inference failure.
    """
    pipe, task, _, _ = get_hf_pipeline(model_id_override=model_id)
    if pipe is None:
        return None

    try:
        # text-generation returns the prompt unless return_full_text=False.
        common_kwargs = {"return_full_text": False} if task == "text-generation" else {}
        outputs = pipe(prompts, **common_kwargs, **generation_kwargs)
        # Pipeline returns list[dict], each with a "generated_text" key (or similar).
        recipes = []
        for o in outputs:
            if isinstance(o, dict):
                txt = o.get("generated_text") or o.get("summary_text") or ""
                if txt:
                    recipes.append(txt.strip())
            elif isinstance(o, str) and o.strip():
                recipes.append(o.strip())
        return [r for r in recipes if r]
    except Exception as exc:
        _HF_LAST_ERROR = f"inference failed: {exc}"
        return None


def generate_hf_completion(prompt: str, model_id: Optional[str] = None, **generation_kwargs) -> Optional[str]:
    """
    Single-shot helper for chat/instruction models; keeps other code simple.
    """
    outputs = generate_hf_recipes([prompt], model_id=model_id, **generation_kwargs)
    if not outputs:
        return None
    return outputs[0]


def get_last_hf_error() -> Optional[str]:
    """Return the last HF load/inference error, if any."""
    return _HF_LAST_ERROR
