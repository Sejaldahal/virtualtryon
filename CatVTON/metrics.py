"""
metrics.py
Flexible FID and IS calculation - works with any number of images
"""

import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np


def load_images(directory, max_images=None, desc="Loading images"):
    """Load images from directory with flexible number"""
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(Path(directory).glob(ext)))
    
    image_paths = sorted(image_paths)
    
    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        print(f"Limiting to first {max_images} images")
    
    print(f"Found {len(image_paths)} images in {directory}")
    
    images = []
    failed = 0
    
    for img_path in tqdm(image_paths, desc=desc):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except Exception as e:
            failed += 1
    
    if failed > 0:
        print(f"  Failed to load {failed} images")
    
    return images


def calculate_metrics(real_images, fake_images, device="cuda"):
    """Calculate FID and IS scores"""
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n❌ No images to process!")
        return None, None, None
    
    # Transform for Inception v3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    
    print("\n📊 Processing real images for FID...")
    for img in tqdm(real_images, desc="Real images"):
        img_tensor = transform(img).unsqueeze(0).to(device)
        fid.update(img_tensor, real=True)
    
    print("\n📊 Processing generated images...")
    for img in tqdm(fake_images, desc="Generated images"):
        img_tensor = transform(img).unsqueeze(0).to(device)
        fid.update(img_tensor, real=False)
        inception_score.update(img_tensor)
    
    # Compute scores
    fid_score = fid.compute()
    is_mean, is_std = inception_score.compute()
    
    return fid_score.item(), is_mean.item(), is_std.item()


def main():
    parser = argparse.ArgumentParser(description="Flexible FID and IS calculation")
    parser.add_argument("--real_dir", required=True, help="Directory with real person images")
    parser.add_argument("--fake_dir", required=True, help="Directory with generated images")
    parser.add_argument("--max_real", type=int, default=None, help="Max real images to use (default: all)")
    parser.add_argument("--max_fake", type=int, default=None, help="Max fake images to use (default: all)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check directories
    real_path = Path(args.real_dir)
    fake_path = Path(args.fake_dir)
    
    if not real_path.exists():
        print(f"❌ Real directory not found: {args.real_dir}")
        return
    
    if not fake_path.exists():
        print(f"❌ Fake directory not found: {args.fake_dir}")
        return
    
    # Load images
    print("\n" + "=" * 70)
    print("🎯 FID & IS Calculation - Flexible Image Count")
    print("=" * 70)
    
    real_images = load_images(args.real_dir, args.max_real, "Loading real images")
    fake_images = load_images(args.fake_dir, args.max_fake, "Loading generated images")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n❌ No images loaded! Please check directories.")
        return
    
    print(f"\n✅ Loaded {len(real_images)} real images")
    print(f"✅ Loaded {len(fake_images)} generated images")
    
    # Use minimum number of images for fair comparison
    num_for_fid = min(len(real_images), len(fake_images))
    
    if num_for_fid < 2:
        print(f"\n⚠️  Need at least 2 images for FID calculation (found {num_for_fid})")
        return
    
    print(f"\n📊 Using {num_for_fid} images for FID calculation")
    
    # Calculate metrics
    fid_score, is_mean, is_std = calculate_metrics(
        real_images[:num_for_fid], 
        fake_images[:num_for_fid], 
        args.device
    )
    
    if fid_score is None:
        return
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Images used: {num_for_fid} pairs")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print("=" * 70)
    
    # Detailed interpretation
    print("\n📈 Detailed Interpretation:")
    print("-" * 50)
    
    # FID interpretation
    print("\nFID Score (Lower is better):")
    if fid_score < 15:
        print(f"  {fid_score:.2f} → Excellent - Very realistic, close to real images")
    elif fid_score < 30:
        print(f"  {fid_score:.2f} → Good - Realistic results with minor artifacts")
    elif fid_score < 50:
        print(f"  {fid_score:.2f} → Moderate - Noticeable artifacts but acceptable")
    else:
        print(f"  {fid_score:.2f} → Poor - Significant artifacts, needs improvement")
    
    # IS interpretation
    print("\nInception Score (Higher is better):")
    if is_mean > 3.5:
        print(f"  {is_mean:.2f} → Excellent - High quality and diverse")
    elif is_mean > 3.0:
        print(f"  {is_mean:.2f} → Good - Good quality and diversity")
    elif is_mean > 2.5:
        print(f"  {is_mean:.2f} → Moderate - Decent quality")
    elif is_mean > 2.0:
        print(f"  {is_mean:.2f} → Fair - Some quality issues")
    else:
        print(f"  {is_mean:.2f} → Poor - Low quality or lack of diversity")
    
    # Confidence based on image count
    print("\nConfidence Level:")
    if num_for_fid >= 200:
        print(f"  {num_for_fid} images → High confidence (statistically significant)")
    elif num_for_fid >= 100:
        print(f"  {num_for_fid} images → Good confidence")
    elif num_for_fid >= 50:
        print(f"  {num_for_fid} images → Moderate confidence")
    else:
        print(f"  {num_for_fid} images → Low confidence (consider using more images)")
    
    print("\n" + "=" * 70)
    print("✨ Calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# python metrics.py \
#     --real_dir ./data/test/image \
#     --fake_dir ./output_fid/generated_images
