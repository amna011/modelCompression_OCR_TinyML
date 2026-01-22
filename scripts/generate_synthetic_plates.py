#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_fonts(font_size: int) -> List[ImageFont.FreeTypeFont]:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    fonts: List[ImageFont.FreeTypeFont] = []
    for path in candidates:
        if os.path.exists(path):
            try:
                fonts.append(ImageFont.truetype(path, font_size))
            except Exception:
                continue
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


def _random_text(rng: random.Random, min_len: int, max_len: int) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(CHARS) for _ in range(length))


def _render_plate(
    text: str,
    size: Tuple[int, int],
    rng: random.Random,
    fonts: List[ImageFont.FreeTypeFont],
) -> Image.Image:
    width, height = size
    bg = rng.randint(210, 255)
    img = Image.new("L", (width, height), color=bg)
    draw = ImageDraw.Draw(img)

    border = rng.randint(2, 6)
    draw.rectangle([border, border, width - border, height - border], outline=0, width=border)

    font = rng.choice(fonts)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, font=font, fill=0)

    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))
    if rng.random() < 0.4:
        angle = rng.uniform(-3.0, 3.0)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic license plates.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=7)
    parser.add_argument("--csv", default="plates.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    fonts = _load_fonts(font_size=int(args.height * 0.55))

    rows: List[str] = ["images,labels"]
    for idx in range(args.count):
        text = _random_text(rng, args.min_len, args.max_len)
        img = _render_plate(text, (args.width, args.height), rng, fonts)
        fname = f"plate_{idx:05d}.png"
        img.save(img_dir / fname)
        rows.append(f"images/{fname},{text}")

    with open(out_dir / args.csv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    print(f"Generated {args.count} plates in {out_dir}")
    print(f"CSV: {out_dir / args.csv}")


if __name__ == "__main__":
    main()
