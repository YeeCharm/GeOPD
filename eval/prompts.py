"""Prompt formatting and image loading utilities for five-benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

# Remote sensing imagery can exceed default PIL decompression limits.
Image.MAX_IMAGE_PIXELS = None

DEFAULT_IMAGE_LONG_EDGE = 1024


def format_prompt(record: dict[str, Any]) -> str:
    """Build the standardized evaluation prompt for a benchmark record."""
    fmt = record.get("format", "mcq")
    if fmt == "multi":
        instruction = (
            "Select all correct options. Return only their letters separated "
            "by commas. Do not explain."
        )
    elif fmt == "mcq":
        instruction = "Select the correct option. Return only its letter. Do not explain."
    else:
        instruction = "Answer using only a concise word, phrase, or number. Do not explain."

    prompt = f"{instruction}\nQuestion: {record['question']}"
    options = record.get("options")
    if options:
        prompt = f"{prompt}\nOptions: {options}"
    return prompt


def load_image(path: str | Path, max_long_edge: int = DEFAULT_IMAGE_LONG_EDGE) -> Image.Image:
    """Load an image in RGB mode and cap its long edge while preserving aspect ratio."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if max_long_edge > 0:
            rgb.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
        return rgb.copy()
