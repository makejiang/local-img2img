# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Apply the mandatory visible disclosure watermark to generated images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


WATERMARK_TEXT = "AI生成"
MIN_IMAGE_SIZE = 128


def validate_image_dimensions(width: int, height: int) -> None:
    """Reject images that cannot carry a readable disclosure watermark."""
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        raise ValueError("图片宽高必须至少为 128 像素")


def _font_candidates() -> tuple[Path, ...]:
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    return fonts_dir / "msyh.ttc", fonts_dir / "simhei.ttf"


def _load_font(font_size: int) -> Any:
    from PIL import ImageFont

    for font_path in _font_candidates():
        if not font_path.is_file():
            continue
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            continue
    raise RuntimeError("未找到支持中文的系统字体，无法添加 AI生成 水印")


def save_ai_generated_png(image: Any, output_path: str | Path) -> None:
    """Add the standard bottom-right watermark and save an RGB PNG."""
    from PIL import Image, ImageDraw

    width, height = image.size
    validate_image_dimensions(width, height)
    short_side = min(width, height)
    font_size = min(48, max(14, round(short_side * 0.04)))
    margin = max(8, round(short_side * 0.015))
    padding_x = max(1, round(font_size * 0.55))
    padding_y = max(1, round(font_size * 0.30))
    radius = max(1, round(font_size * 0.35))

    font = _load_font(font_size)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (0, 0), WATERMARK_TEXT, font=font
    )
    text_width = text_right - text_left
    text_height = text_bottom - text_top

    box_right = width - margin - 1
    box_bottom = height - margin - 1
    box_left = box_right - text_width - 2 * padding_x
    box_top = box_bottom - text_height - 2 * padding_y
    draw.rounded_rectangle(
        (box_left, box_top, box_right, box_bottom),
        radius=radius,
        fill=(0, 0, 0, 153),
    )
    draw.text(
        (box_left + padding_x - text_left, box_top + padding_y - text_top),
        WATERMARK_TEXT,
        font=font,
        fill=(255, 255, 255, 230),
    )

    watermarked = Image.alpha_composite(base, overlay).convert("RGB")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        watermarked.save(temporary_path, format="PNG")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
