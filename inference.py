"""Run AdaIN neural style transfer with the included pretrained decoder.

Example:
    python inference.py --content content_data/lenna.jpg \
        --style style_data/mondrian.jpg --output outputs/lenna_mondrian.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalization


def parse_args():
    parser = argparse.ArgumentParser(description="AdaIN Neural Style Transfer inference")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--output", default="outputs/stylized_output.png", help="Output image path")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Style strength from 0 (content) to 1 (full style), default: 1")
    parser.add_argument("--size", type=int, default=256,
                        help="Square inference resolution (try 256 on CPU; default: 256)")
    parser.add_argument("--decoder", default="weights/decoder_final.pth",
                        help="Path to pretrained decoder checkpoint")
    parser.add_argument("--vgg", default="vgg_normalised.pth",
                        help="Path to pretrained VGG encoder weights")
    return parser.parse_args()


def load_image(path: str, size: int, device: torch.device) -> torch.Tensor:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    # Match the square image dimensions used by this project's training pipeline.
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    import numpy as np
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.size < 64:
        raise ValueError("--size must be at least 64")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = VGGEncoder(args.vgg).to(device).eval()
    decoder = Decoder().to(device)
    checkpoint_path = Path(args.decoder)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases
        state = torch.load(checkpoint_path, map_location="cpu")
    decoder.load_state_dict(state, strict=True)
    decoder.eval()

    content = load_image(args.content, args.size, device)
    style = load_image(args.style, args.size, device)

    with torch.inference_mode():
        content_features = encoder(content, is_test=True)
        style_features = encoder(style, is_test=True)
        target_features = adaptive_instance_normalization(content_features, style_features)
        mixed_features = args.alpha * target_features + (1.0 - args.alpha) * content_features
        output = decoder(mixed_features).squeeze(0).cpu().clamp(0, 1)

    import numpy as np
    output_image = Image.fromarray((output.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"Stylized image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
