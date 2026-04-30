import os
import re
import sys
import random
import platform
import argparse
from datetime import date, timedelta
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

#download fonts

FONT_URLS = {
    "Caveat": (
        "https://github.com/googlefonts/caveat/raw/main/fonts/ttf/Caveat-Regular.ttf"
    ),
    "Cedarville": (
        "https://github.com/google/fonts/raw/refs/heads/main/ofl/cedarvillecursive/Cedarville-Cursive.ttf"
    ),
    "Dawning": (
        "https://github.com/google/fonts/raw/refs/heads/main/ofl/dawningofanewday/DawningofaNewDay.ttf"
    ),
    "Saint Delafield": (
        "https://github.com/google/fonts/raw/refs/heads/main/ofl/mrssaintdelafield/MrsSaintDelafield-Regular.ttf"
    ),
    "Rainbow": (
        "https://github.com/google/fonts/raw/refs/heads/main/ofl/overtherainbow/OvertheRainbow.ttf"
    ),
}


def download_fonts(output_dir: str = "fonts") -> list[str]:
    """Download all handwriting fonts from GitHub and return their local paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for name, url in FONT_URLS.items():
        path = os.path.join(output_dir, f"{name.replace(' ', '_')}.ttf")

        if os.path.exists(path):
            print(f"  [skip] {name} already exists.")
            paths.append(path)
            continue

        print(f"  [download] {name} ...")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"    Saved → {path}")
            paths.append(path)
        except Exception as e:
            print(f"    WARNING: Could not download '{name}': {e}")

    if not paths:
        raise RuntimeError(
            "No fonts were downloaded. Check your internet connection."
        )

    return paths


#generate date string

def _no_pad(code: str) -> str:
    """Return OS-appropriate no-zero-padding format code."""
    return f"%#{code}" if platform.system() == "Windows" else f"%-{code}"


def build_date_formats() -> list[str]:
    m, d = _no_pad("m"), _no_pad("d")
    return [
        "%m/%d/%Y",          # 04/29/2026
        "%m/%d/%y",          # 04/29/26
        "%d/%m/%Y",          # 29/04/2026
        "%d/%m/%y",          # 29/04/26
        f"{m}/{d}/%Y",       # 4/29/2026
        f"{m}/{d}/%y",       # 4/29/26
        "%B %d, %Y",         # April 29, 2026
        f"%B {d}, %Y",       # April 29, 2026 (no zero-pad day)
        f"{d} %B %Y",        # 29 April 2026
        f"{d} %b %Y",        # 29 Apr 2026
        "%b %d, %Y",         # Apr 29, 2026
        "%m-%d-%Y",          # 04-29-2026
        "%d-%m-%Y",          # 29-04-2026
        f"{m}.{d}.%Y",       # 4.29.2026
        "%m.%d.%Y",          # 04.29.2026
        "%Y-%m-%d",          # 2026-04-29  (ISO)
        "%d %b '%y",         # 29 Apr '26
    ]


DATE_FORMATS = build_date_formats()


def random_date(start_year: int = 1990, end_year: int = 2030) -> date:
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def random_date_string() -> str:
    fmt = random.choice(DATE_FORMATS)
    return random_date().strftime(fmt)


#generate image
INK_PALETTES = [
    # Dark blue-black ballpoint
    ((0, 30),  (0, 30),  (40, 110)),
    # True black pen
    ((0, 25),  (0, 25),  (0,  25)),
    # Dark navy
    ((0, 20),  (0, 30),  (60, 130)),
    # Faded/old ink (slightly brownish)
    ((30, 70), (20, 50), (10, 40)),
    # Pencil (mid-gray)
    ((80, 130),(80, 130),(80, 130)),
]

# Paper background palettes
PAPER_PALETTES = [
    # Clean white paper
    ((240, 255), (240, 255), (235, 255)),
    # Aged/yellowed paper
    ((225, 245), (215, 235), (185, 210)),
    # Light blue ruled paper
    ((230, 245), (235, 248), (245, 255)),
    # Off-white notepad
    ((235, 252), (232, 250), (220, 242)),
]


def random_color(palette: tuple) -> tuple:
    return tuple(random.randint(lo, hi) for lo, hi in palette)


def render_date(
    date_str: str,
    font_paths: list[str],
    canvas_size: tuple[int, int] = (300, 80),
) -> Image.Image:
    """Render a date string onto a paper-like canvas."""

    # Paper background
    bg = random_color(random.choice(PAPER_PALETTES))
    img = Image.new("RGB", canvas_size, color=bg)
    draw = ImageDraw.Draw(img)

    # Font
    font_path = random.choice(font_paths)
    font_size = random.randint(22, 38)
    font      = ImageFont.truetype(font_path, font_size)

    # Measure text
    bbox   = draw.textbbox((0, 0), date_str, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Random position with padding guard
    pad_x = 8
    pad_y = 5
    max_x = max(pad_x, canvas_size[0] - text_w - pad_x)
    max_y = max(pad_y, canvas_size[1] - text_h - pad_y)
    x = random.randint(pad_x, max_x)
    y = random.randint(pad_y, max_y)

    # Ink color
    ink = random_color(random.choice(INK_PALETTES))
    draw.text((x, y), date_str, font=font, fill=ink)

    return img


#augment image

def augment(img: Image.Image) -> Image.Image:
    """Apply realistic handwriting-scan augmentations."""

    bg_color = img.getpixel((0, 0))

    # 1. Slight rotation (-5° to +5°)
    angle = random.uniform(-5, 5)
    img   = img.rotate(angle, expand=False, fillcolor=bg_color)

    # 2. Gaussian noise (scanner grain / paper texture)
    noise_std = random.uniform(2, 10)
    arr   = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, noise_std, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    # 3. Ink blur (feathering / out-of-focus scan) — applied 40% of the time
    if random.random() < 0.4:
        radius = random.uniform(0.3, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 4. Brightness variation (lighting during scanning)
    factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 5. Contrast variation (ink density)
    factor = random.uniform(0.9, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 6. Slight sharpening — applied 30% of the time
    if random.random() < 0.3:
        img = img.filter(ImageFilter.SHARPEN)

    return img


#generate dataset 

def generate_dataset(
    n: int          = 500,
    output_dir: str = "output",
    font_dir: str   = "fonts",
    seed: int       = None,
) -> None:
    """
    Generate n synthetic handwritten date images.

    Output structure:
        output/
            images/
                00000.png
                00001.png
                ...
            labels.tsv      (filename <TAB> date_string)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Download fonts
    print("\n=== Downloading fonts ===")
    font_paths = download_fonts(font_dir)
    print(f"Using {len(font_paths)} font(s).\n")

    # Prepare output folders
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels = []
    print(f"=== Generating {n} images ===")

    for i in range(n):
        date_str = random_date_string()
        img      = render_date(date_str, font_paths)
        img      = augment(img)

        filename = f"{i:05d}.png"
        img.save(os.path.join(images_dir, filename))
        labels.append(f"{filename}\t{date_str}")

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} images generated...")

    # Save labels
    labels_path = os.path.join(output_dir, "labels.tsv")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("filename\tdate_string\n")
        f.write("\n".join(labels))

    print(f"\n=== Done ===")
    print(f"Images : {images_dir}")
    print(f"Labels : {labels_path}")
    print(f"Total  : {n} images across {len(DATE_FORMATS)} date formats\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic handwritten date generator")
    parser.add_argument("--n",      type=int, default=500,    help="Number of images to generate")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fonts",  type=str, default="fonts",  help="Font cache directory")
    parser.add_argument("--seed",   type=int, default=None,   help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(
        n          = args.n,
        output_dir = args.output,
        font_dir   = args.fonts,
        seed       = args.seed,
    )
