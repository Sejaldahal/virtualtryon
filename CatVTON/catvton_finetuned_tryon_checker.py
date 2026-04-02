"""
CatVTON Optimized Try-On — With Fine-tuned Attention Weights

Usage:
    python3 catvton_finetuned_tryon_checker.py \
        --person  person.jpg \
        --cloth   cloth.jpg \
        --output  epoch_5_result.png \
        --attention_ckpt catvton_attn_layers_epoch_5.pt
        --steps 25 \
        --repaint
"""

import argparse
import os
import gc
import sys
import numpy as np
import torch
from PIL import Image

# Add CatVTON to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from model.utils import get_trainable_module
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    p = argparse.ArgumentParser(description="CatVTON Virtual Try-On with Fine-tuned Attention")
    
    # Required arguments
    p.add_argument("--person", required=True, help="Path to person image")
    p.add_argument("--cloth", required=True, help="Path to garment image")
    p.add_argument("--output", default="tryon_result.png", help="Output path")
    
    # Model paths
    p.add_argument("--base_model_path", default="booksforcharlie/stable-diffusion-inpainting")
    p.add_argument("--resume_path", default=None, help="Path to CatVTON model (auto-downloads if not provided)")
    p.add_argument("--attention_ckpt", default="catvton_attn_layers_epoch_1.pt", 
                   help="Path to your fine-tuned attention weights")
    
    # Resolution handling (auto by default)
    p.add_argument("--width", type=int, default=None, help="Width (auto-detected if not specified)")
    p.add_argument("--height", type=int, default=None, help="Height (auto-detected if not specified)")
    p.add_argument("--orientation", choices=["auto", "portrait", "landscape", "square"], 
                   default="auto", help="Output orientation (default: auto)")
    p.add_argument("--max_size", type=int, default=1024, 
                   help="Maximum dimension for auto-resize (default: 1024)")
    
    # Generation parameters
    p.add_argument("--steps", type=int, default=25, help="Diffusion steps")
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--cloth_type", default="upper", choices=["upper", "lower", "overall"],
                   help="Type of garment for masking")
    
    # Background options
    p.add_argument("--repaint", action="store_true", 
                   help="Keep original background (paste result back on original)")
    p.add_argument("--background_replace", type=str, default=None,
                   help="Path to background image to replace the original background")
    p.add_argument("--background_color", type=str, default=None,
                   help="Solid color background: 'R,G,B' e.g., '255,255,255' for white")
    p.add_argument("--background_blur", type=int, default=None,
                   help="Blur background (keep person, blur surroundings) - kernel size e.g., 21")
    p.add_argument("--remove_background", action="store_true",
                   help="Remove background completely (transparent PNG output)")
    
    # Performance options
    p.add_argument("--allow_tf32", action="store_true", default=True)
    p.add_argument("--no_xformers", action="store_true", help="Disable xformers")
    
    return p.parse_args()


def load_finetuned_attention(pipeline, attention_ckpt_path, device):
    """Load your fine-tuned attention weights into the pipeline"""
    
    print(f"\n🔧 Loading fine-tuned attention weights from: {attention_ckpt_path}")
    
    if not os.path.exists(attention_ckpt_path):
        print(f"❌ Attention weights not found: {attention_ckpt_path}")
        return False
    
    try:
        # Load safetensors file
        attention_weights = torch.load(attention_ckpt_path, map_location=device)
        print(f"✅ Loaded {len(attention_weights)} attention keys")
        
        # Get attention modules from pipeline
        attn_modules = get_trainable_module(pipeline.unet, "attention")
        print(f"   Found {len(attn_modules)} attention modules in pipeline")
        
        # Load weights into attention modules
        loaded_count = 0
        
        # Try to match keys
        for i, module in enumerate(attn_modules):
            module_state = module.state_dict()
            
            # Try different key formats
            for key in module_state.keys():
                # Try various possible key names from the checkpoint
                possible_keys = [
                    key,
                    f"attention_{i}.{key}",
                    f"attn_{i}.{key}",
                    f"down_blocks.{i//2}.attentions.{i%2}.{key}",
                ]
                
                for pkey in possible_keys:
                    if pkey in attention_weights:
                        module_state[key] = attention_weights[pkey].to(device)
                        loaded_count += 1
                        break
            
            # Load updated state
            module.load_state_dict(module_state, strict=False)
        
        print(f"   Loaded {loaded_count} weight entries into attention modules")
        
        # Move to device
        pipeline.unet = pipeline.unet.to(device)
        
        print("✅ Fine-tuned attention weights loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading attention weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_densepose_files(model_path):
    """Download DensePose config files if they don't exist"""
    densepose_dir = os.path.join(model_path, "DensePose")
    os.makedirs(densepose_dir, exist_ok=True)
    
    config_file = os.path.join(densepose_dir, "densepose_rcnn_R_50_FPN_s1x.yaml")
    
    if not os.path.exists(config_file):
        print("📥 Downloading DensePose config files...")
        repo_url = "https://huggingface.co/spaces/yisol/CatVTON/resolve/main/DensePose/"
        files = [
            "densepose_rcnn_R_50_FPN_s1x.yaml",
            "DensePose_ResNet50_FPN_s1x-e2e.pkl"
        ]
        
        for file in files:
            file_path = os.path.join(densepose_dir, file)
            if not os.path.exists(file_path):
                try:
                    import requests
                    url = repo_url + file
                    print(f"   Downloading {file}...")
                    response = requests.get(url)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"   Warning: Could not download {file}: {e}")
    
    return densepose_dir


def auto_detect_resolution(person_image, orientation="auto", max_size=1024):
    """
    Automatically determine optimal resolution based on image aspect ratio
    """
    orig_w, orig_h = person_image.size
    aspect_ratio = orig_w / orig_h
    
    print(f"\n📐 Auto-detecting optimal resolution...")
    print(f"   Original: {orig_w}x{orig_h} (ratio: {aspect_ratio:.2f})")
    
    # Determine orientation
    if orientation == "auto":
        if aspect_ratio > 1.2:
            orientation = "landscape"
            print(f"   Detected: Landscape image")
        elif aspect_ratio < 0.8:
            orientation = "portrait"
            print(f"   Detected: Portrait image")
        else:
            orientation = "square"
            print(f"   Detected: Square image")
    
    # Set target dimensions based on orientation
    if orientation == "portrait":
        target_h = max_size
        target_w = int(max_size * 0.75)
    elif orientation == "landscape":
        target_w = max_size
        target_h = int(max_size * 0.75)
    else:
        target_w = max_size
        target_h = max_size
    
    # Round to multiples of 64
    target_w = (target_w // 64) * 64
    target_h = (target_h // 64) * 64
    
    print(f"   Target: {target_w}x{target_h} ({orientation})")
    
    return target_w, target_h, orientation


def replace_background(person_image, result_image, mask, background=None, color=None):
    """Replace the background of the result image"""
    print("\n🎨 Replacing background...")
    
    if mask.size != result_image.size:
        mask = mask.resize(result_image.size, Image.Resampling.NEAREST)
    
    mask_np = np.array(mask.convert("L"))
    mask_bin = (mask_np > 128).astype(np.uint8)
    
    if background is not None:
        bg_image = Image.open(background).convert("RGB")
        bg_image = bg_image.resize(result_image.size, Image.Resampling.LANCZOS)
        bg_np = np.array(bg_image)
        print(f"   Using background image: {background}")
    elif color is not None:
        color_tuple = tuple(map(int, color.split(',')))
        bg_np = np.full((result_image.height, result_image.width, 3), color_tuple, dtype=np.uint8)
        print(f"   Using solid color: RGB{color_tuple}")
    else:
        raise ValueError("Either background image or color must be provided")
    
    result_np = np.array(result_image)
    final_np = result_np * mask_bin[:, :, None] + bg_np * (1 - mask_bin[:, :, None])
    final_image = Image.fromarray(final_np.astype(np.uint8))
    
    print("   Background replaced successfully")
    return final_image


def blur_background(person_image, result_image, mask, blur_kernel=21):
    """Apply blur to background while keeping person sharp"""
    print(f"\n🌀 Blurring background (kernel size: {blur_kernel})...")
    
    from PIL import ImageFilter
    
    if mask.size != result_image.size:
        mask = mask.resize(result_image.size, Image.Resampling.NEAREST)
    
    blurred = result_image.filter(ImageFilter.GaussianBlur(radius=blur_kernel // 2))
    
    mask_np = np.array(mask.convert("L"))
    mask_bin = (mask_np > 128).astype(np.uint8)
    
    result_np = np.array(result_image)
    blurred_np = np.array(blurred)
    final_np = result_np * mask_bin[:, :, None] + blurred_np * (1 - mask_bin[:, :, None])
    final_image = Image.fromarray(final_np.astype(np.uint8))
    
    print("   Background blurred successfully")
    return final_image


def remove_background_transparent(result_image, mask):
    """Make background transparent (PNG with alpha channel)"""
    print("\n🔲 Removing background (creating transparent PNG)...")
    
    if mask.size != result_image.size:
        mask = mask.resize(result_image.size, Image.Resampling.NEAREST)
    
    result_rgba = result_image.convert("RGBA")
    mask_np = np.array(mask.convert("L"))
    
    data = np.array(result_rgba)
    data[:, :, 3] = mask_np
    final_image = Image.fromarray(data, 'RGBA')
    
    print("   Background removed (transparent PNG created)")
    return final_image


def clear_memory():
    """Aggressively clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    
    # Set environment variable for memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Determine model path
    if args.resume_path is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--zhengchong--CatVTON/snapshots")
        if os.path.exists(cache_dir):
            snapshots = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
            if snapshots:
                args.resume_path = os.path.join(cache_dir, snapshots[0])
                print(f"\n📁 Using cached model: {args.resume_path}")
            else:
                print("\n📁 Model will be downloaded automatically...")
                args.resume_path = "zhengchong/CatVTON"
        else:
            args.resume_path = "zhengchong/CatVTON"
    
    # Download DensePose files if needed
    if os.path.exists(args.resume_path) and not args.resume_path.startswith("zhengchong"):
        download_densepose_files(args.resume_path)
    
    # Check device
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU will be extremely slow.")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {vram_gb:.1f}GB")
    
    # Load original image to detect resolution
    original_person_image = Image.open(args.person).convert("RGB")
    
    # Auto-detect or use user-specified resolution
    if args.width is None or args.height is None:
        target_w, target_h, orientation = auto_detect_resolution(
            original_person_image, args.orientation, args.max_size
        )
    else:
        target_w, target_h = args.width, args.height
        print(f"\n📐 Using user-specified resolution: {target_w}x{target_h}")
    
    print(f"\n📁 Input files:")
    print(f"   Person: {args.person}")
    print(f"   Cloth: {args.cloth}")
    print(f"   Output: {args.output}")
    if args.attention_ckpt:
        print(f"   Fine-tuned Attention: {args.attention_ckpt}")
    
    # Clear memory before loading models
    clear_memory()
    
    # Initialize weight dtype
    print("\n🔧 Loading CatVTON pipeline...")
    weight_dtype = init_weight_dtype(args.mixed_precision)
    
    # For RTX 2080 Ti, bf16 is not supported
    if args.mixed_precision == "bf16" and "2080" in gpu_name:
        print("⚠️  RTX 2080 Ti does NOT support bf16. Switching to fp16...")
        args.mixed_precision = "fp16"
        weight_dtype = torch.float16
    
    # Load pipeline with skip_safety_check to avoid NSFW blocking
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=args.resume_path,
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        use_tf32=args.allow_tf32,
        device=device,
        skip_safety_check=True,  # Disable safety checker
    )
    
    # Load your fine-tuned attention weights
    if args.attention_ckpt and os.path.exists(args.attention_ckpt):
        load_finetuned_attention(pipeline, args.attention_ckpt, device)
    else:
        print(f"\n⚠️ Attention weights not found: {args.attention_ckpt}")
        print("   Using original CatVTON attention weights")
    
    # Load AutoMasker with proper path
    print("\n🔍 Loading AutoMasker...")
    
    if os.path.exists(args.resume_path) and not args.resume_path.startswith("zhengchong"):
        densepose_path = args.resume_path
    else:
        local_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        densepose_path = local_cache
        os.makedirs(densepose_path, exist_ok=True)
        download_densepose_files(densepose_path)
    
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(densepose_path, "DensePose"),
        schp_ckpt=os.path.join(args.resume_path if os.path.exists(args.resume_path) else densepose_path, "SCHP"),
        device=device,
    )
    
    # Load and preprocess images
    print("🖼️  Preprocessing images...")
    cloth_image = Image.open(args.cloth).convert("RGB")
    
    person_resized = resize_and_crop(original_person_image.copy(), (target_w, target_h))
    cloth_resized = resize_and_padding(cloth_image, (target_w, target_h))
    
    # Generate mask on the resized image
    print(f"🎭 Generating mask for {args.cloth_type} garment...")
    mask_result = automasker(person_resized, args.cloth_type)
    mask = mask_result["mask"]
    
    # Save mask for debugging
    mask.save("debug_mask.png")
    print(f"💾 Mask saved to debug_mask.png")
    
    # Run inference
    print(f"🎨 Running diffusion ({args.steps} steps) with fine-tuned attention...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    result = pipeline(
        image=person_resized,
        condition_image=cloth_resized,
        mask=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    
    # Handle different return types
    if isinstance(result, tuple) or isinstance(result, list):
        result_image = result[0]
    elif hasattr(result, 'images'):
        result_image = result.images[0]
    else:
        result_image = result
    
    # Apply background modifications
    final_image = result_image
    
    if args.repaint:
        print("\n🖌️  Repainting onto original background...")
        orig_w, orig_h = original_person_image.size
        
        result_resized = result_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        mask_resized = mask.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        
        mask_np = np.array(mask_resized.convert("L"))
        mask_bin = (mask_np > 128).astype(np.uint8)
        result_np = np.array(result_resized)
        person_np = np.array(original_person_image)
        
        final_image = Image.fromarray(
            (person_np * (1 - mask_bin[:, :, None]) + result_np * mask_bin[:, :, None]).astype(np.uint8)
        )
        print("   Repainting completed")
    
    if args.background_replace:
        final_image = replace_background(
            original_person_image, final_image, mask,
            background=args.background_replace, color=None
        )
    elif args.background_color:
        final_image = replace_background(
            original_person_image, final_image, mask,
            background=None, color=args.background_color
        )
    
    if args.background_blur:
        final_image = blur_background(
            original_person_image, final_image, mask, args.background_blur
        )
    
    if args.remove_background:
        final_image = remove_background_transparent(final_image, mask)
    
    # Save result
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    
    if args.remove_background and not args.output.lower().endswith('.png'):
        output_path = args.output.rsplit('.', 1)[0] + '.png'
        print(f"\n⚠️  Changed output format to PNG for transparency support")
    else:
        output_path = args.output
    
    final_image.save(output_path)
    print(f"\n✅ Result saved → {output_path}")
    
    # Save comparison strip
    comp_path = output_path.replace(".png", "_comparison.png").replace(".jpg", "_comparison.png")
    comparison = Image.new("RGB", (target_w * 3 + 20, target_h), (230, 230, 230))
    comparison.paste(person_resized, (0, 0))
    comparison.paste(cloth_resized, (target_w + 10, 0))
    
    result_for_compare = final_image
    if final_image.size != (target_w, target_h):
        if final_image.mode == 'RGBA':
            rgb_version = Image.new('RGB', final_image.size, (255, 255, 255))
            rgb_version.paste(final_image, mask=final_image.split()[3] if final_image.mode == 'RGBA' else None)
            result_for_compare = rgb_version
        result_for_compare = result_for_compare.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    comparison.paste(result_for_compare, (target_w * 2 + 20, 0))
    comparison.save(comp_path)
    print(f"✅ Comparison saved → {comp_path}")
    
    print("\n📊 Summary:")
    print(f"   Resolution: {target_w}x{target_h}")
    print(f"   Steps: {args.steps}")
    print(f"   Cloth type: {args.cloth_type}")
    if args.attention_ckpt and os.path.exists(args.attention_ckpt):
        print(f"   Attention: Using fine-tuned weights from {os.path.basename(args.attention_ckpt)}")
    else:
        print(f"   Attention: Original CatVTON")
    if args.repaint:
        print(f"   Background: Original (repainted)")
    elif args.background_replace:
        print(f"   Background: Replaced with image")
    elif args.background_color:
        print(f"   Background: Solid color")
    elif args.background_blur:
        print(f"   Background: Blurred")
    elif args.remove_background:
        print(f"   Background: Transparent")
    else:
        print(f"   Background: Model-generated")
    
    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()