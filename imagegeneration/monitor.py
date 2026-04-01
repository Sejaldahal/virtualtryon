
"""
Training Monitor & Validation Script
Generates sample images during training to monitor quality
"""

import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

def generate_validation_images(lora_checkpoint_path, prompts, output_dir, epoch_num,negative_prompt=None):
    """
    Generate sample images from validation prompts
    
    Args:
        lora_checkpoint_path: Path to LoRA checkpoint
        prompts: List of text prompts to test
        output_dir: Where to save generated images
        epoch_num: Current epoch number
    """
    
    print(f"\n🎨 Generating validation images for epoch {epoch_num}...")
    
    # Create output directory for this epoch
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch_num}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None  # Disable for faster generation
    )
    
    # Load LoRA weights using PEFT (recommended). If that fails, try diffusers' loader.
    try:
        # Apply PEFT adapter to the loaded UNet
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_checkpoint_path)
        pipe.unet.to(torch.float16)
        print(f"✅ Loaded LoRA adapter via PEFT from {lora_checkpoint_path}")
    except Exception as e_peft:
        print(f"⚠️  PEFT load failed: {e_peft}")
        # Try diffusers-style LoRA weights if available
        if hasattr(pipe, "load_lora_weights"):
            try:
                pipe.load_lora_weights(lora_checkpoint_path)
                print(f"✅ Loaded LoRA weights via diffusers' load_lora_weights from {lora_checkpoint_path}")
            except Exception as e_diff:
                print(f"❌ Fallback load_lora_weights also failed: {e_diff}")
                return
        else:
            print("❌ No supported fallback loader found (no load_lora_weights).")
            return

    pipe = pipe.to("cuda")
    
    # Generate images
    generated_images = []
    
    for i, prompt in enumerate(prompts):
        print(f"  Generating {i+1}/{len(prompts)}: {prompt}")
        
        try:
            # Generate image
            with torch.no_grad():
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,  # Fewer steps for faster validation
                    guidance_scale=7.5
                ).images[0]
            
            # Save individual image
            img_path = os.path.join(epoch_dir, f"prompt_{i+1}.png")
            image.save(img_path)
            generated_images.append((prompt, image))
            
        except Exception as e:
            print(f"❌ Error generating image for prompt '{prompt}': {e}")
    
    # Create comparison grid
    if generated_images:
        create_validation_grid(generated_images, epoch_dir, epoch_num)
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    print(f"✅ Validation images saved to {epoch_dir}\n")

def create_validation_grid(images_with_prompts, output_dir, epoch_num):
    """Create a grid of validation images with prompts"""
    
    num_images = len(images_with_prompts)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for idx, (prompt, image) in enumerate(images_with_prompts):
        axes[idx].imshow(image)
        axes[idx].axis('off')
        # Wrap long prompts
        wrapped_prompt = '\n'.join([prompt[i:i+30] for i in range(0, len(prompt), 30)])
        axes[idx].set_title(wrapped_prompt, fontsize=8)
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"validation_grid_epoch_{epoch_num}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Grid saved: {grid_path}")

def monitor_training_progress(output_dir):
    """
    Monitor training progress by checking saved checkpoints
    and generating validation images
    """
    
    validation_prompts = [
        "a red summer dress with floral patterns",
        "blue denim jeans with ripped knees",
        "black leather jacket with zippers",
        "white cotton t-shirt with round neck"
    ]
    
    print("👀 Monitoring training progress...")
    print("Looking for checkpoints in:", output_dir)
    
    # Look for checkpoint directories
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))]
    
    if not checkpoints:
        print("⚠️  No checkpoints found yet. Training may still be in progress.")
        return
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Create validation output directory
    validation_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Generate validation images for each checkpoint
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        epoch_num = int(checkpoint.split('-')[1])
        
        # Check if validation already done
        epoch_validation_dir = os.path.join(validation_dir, f"epoch_{epoch_num}")
        if os.path.exists(epoch_validation_dir):
            print(f"⏭️  Skipping epoch {epoch_num} (already validated)")
            continue
        
        generate_validation_images(
            checkpoint_path,
            validation_prompts,
            validation_dir,
            epoch_num
        )

def compare_checkpoints(checkpoint_paths, prompt, negative_prompt, output_path="comparison.png"):
    """
    Compare multiple LoRA checkpoints side by side (CORRECT VERSION)
    """

    print(f"\n🔍 Comparing {len(checkpoint_paths)} checkpoints...")
    print(f"Prompt: {prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    images = []
    labels = []

    for checkpoint_path in checkpoint_paths:
        epoch_label = (
            checkpoint_path.split("/")[-1]
            if "checkpoint-" in checkpoint_path
            else "final"
        )

        try:
            print(f"✅ Loading {epoch_label}")

            # 1️⃣ Load base Stable Diffusion model
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None
            ).to(device)

            # 2️⃣ Load LoRA weights (CORRECT way)
            try:
                # Preferred: diffusers LoRA loader
                pipe.load_lora_weights(checkpoint_path)
                pipe.fuse_lora()
            except Exception:
                # Fallback: PEFT loader
                pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_path)
                pipe.unet = pipe.unet.to(dtype)

            generator = torch.Generator(device=device).manual_seed(42)

            # 3️⃣ Generate image
            with torch.no_grad():
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator

                ).images[0]

            images.append(image)
            labels.append(epoch_label)

            del pipe
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error with checkpoint {checkpoint_path}: {e}")

    # 4️⃣ Create comparison grid
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        if len(images) == 1:
            axes = [axes]

        for idx, (image, label) in enumerate(zip(images, labels)):
            axes[idx].imshow(image)
            axes[idx].axis("off")
            axes[idx].set_title(label, fontsize=12, fontweight="bold")

        plt.suptitle(f"Prompt: {prompt}", fontsize=10)
        plt.tight_layout()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Comparison saved to {output_path}")


#==================== USAGE EXAMPLES ====================
if __name__ == "__main__":

    
    # OPTION 3: Generate validation images for specific checkpoint
    generate_validation_images(
        lora_checkpoint_path="comparelora_model/checkpoint-12",
        prompts=[
            "<cloth> the basic t - shirt in grey, round neck, short sleeves,on white background",
         
        ],
        output_dir="lora_model/check_images",
        epoch_num=12,
       negative_prompt = (
                "person, human, model, mannequin, body, torso, arms, hands, "
                "face, neck, shoulders, legs,textured background, cluttered background, shadow"
            )

    )




