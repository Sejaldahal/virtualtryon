"""
Text-to-Outfit Generation Training Pipeline using LoRA
Fine-tunes Stable Diffusion on clothing dataset with BLIP captions
"""

# First, install required packages:
# pip install diffusers transformers accelerate torch torchvision peft datasets pillow

import os
import torch
# ✅ Enable cuDNN (default True, but good to set explicitly)
torch.backends.cudnn.enabled = True

# ✅ Allow cuDNN to pick the fastest convolution algorithm for your GPU & fixed image size
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler,AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ✅ Enable cuDNN (default True, but good to set explicitly)
torch.backends.cudnn.enabled = True

# ✅ Allow cuDNN to pick the fastest convolution algorithm for your GPU & fixed image size
torch.backends.cudnn.benchmark = True


# ==================== CONFIGURATION ====================
class Config:
    # Paths
    IMAGE_DIR = "/home/sejal/AI Virtual Tryon/datasets/train/cloth"  # Folder with your images
    CAPTION_FILE = "/home/sejal/AI Virtual Tryon/datasets/train/captions.json"  # JSON file with image-caption pairs
    OUTPUT_DIR = "/home/sejal/AI Virtual Tryon/diffusionmodel/comparelora_model"  # Where to save trained model
    LOGS_DIR = "/home/sejal/AI Virtual Tryon/diffusionmodel/comparetraining_logs"  # Where to save metrics
    
    # Model settings
    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"  # Base model
    RESOLUTION = 512  # Image resolution (512 for SD1.5)
    
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # if memory allows
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4

    SAVE_EVERY = 1  # every epoch

    VALIDATION_PROMPTS = [
    "<cloth> a red summer dress with floral patterns",
    "<cloth> blue denim jeans with ripped knees",
    "<cloth> black leather jacket with zippers"
    ]


    
    # LoRA settings - Increased for larger dataset
    LORA_RANK = 8  # Higher rank for 11K images dataset
    LORA_ALPHA = 16  # Scaled with rank
    
    # Mixed precision (fp16 recommended for RTX 2080 Ti)
    MIXED_PRECISION = "fp16"
    
    # Optimizer settings
    USE_8BIT_ADAM = True  # Saves memory, install: pip install bitsandbytes

# ==================== METRICS & LOGGING ====================
class MetricsTracker:
    """Track training metrics and save visualizations"""
    
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        
        self.step_losses = []  # Loss per step
        self.epoch_losses = []  # Average loss per epoch
        self.epoch_nums = []
        self.global_steps = []
        self.current_step = 0
        self.metrics_dict = {}
    
    def log_step(self, loss, global_step):
        """Log loss for current step"""
        self.step_losses.append(loss)
        self.global_steps.append(global_step)
        self.current_step = global_step
    
    def log_epoch(self, epoch, avg_loss):
        """Log metrics for completed epoch"""
        self.epoch_losses.append(avg_loss)
        self.epoch_nums.append(epoch + 1)
        self.metrics_dict[f"epoch_{epoch+1}"] = {
            "avg_loss": avg_loss,
            "timestamp": datetime.now().isoformat()
        }
    
    def plot_loss_curve(self, filename="loss_curve.png"):
        """Plot and save loss curve"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Step-wise loss
        axes[0].plot(self.global_steps, self.step_losses, alpha=0.6, label="Step Loss")
        axes[0].set_xlabel("Global Step")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Training Loss per Step")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Epoch-wise average loss
        axes[1].plot(self.epoch_nums, self.epoch_losses, marker='o', linewidth=2, markersize=8, label="Epoch Avg Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Average Loss (MSE)")
        axes[1].set_title("Average Loss per Epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.logs_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Loss curve saved to {save_path}")
    
    def save_metrics_json(self, filename="metrics.json"):
        """Save metrics to JSON file"""
        full_metrics = {
            "training_summary": {
                "total_steps": self.current_step,
                "total_epochs": len(self.epoch_losses),
                "final_epoch_loss": self.epoch_losses[-1] if self.epoch_losses else None,
                "min_epoch_loss": min(self.epoch_losses) if self.epoch_losses else None,
                "max_epoch_loss": max(self.epoch_losses) if self.epoch_losses else None,
                "timestamp": datetime.now().isoformat()
            },
            "epoch_metrics": self.metrics_dict,
            "step_metrics": {
                "min_loss": float(np.min(self.step_losses)) if self.step_losses else None,
                "max_loss": float(np.max(self.step_losses)) if self.step_losses else None,
                "avg_loss": float(np.mean(self.step_losses)) if self.step_losses else None,
                "std_loss": float(np.std(self.step_losses)) if self.step_losses else None
            }
        }
        
        save_path = os.path.join(self.logs_dir, filename)
        with open(save_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)
        print(f"📋 Metrics saved to {save_path}")
    
    def print_summary(self):
        """Print training summary"""
        if not self.epoch_losses:
            return
        
        print("\n" + "="*50)
        print("📈 TRAINING SUMMARY")
        print("="*50)
        print(f"Total Epochs: {len(self.epoch_losses)}")
        print(f"Total Steps: {self.current_step}")
        print(f"Final Loss: {self.epoch_losses[-1]:.6f}")
        print(f"Best Loss: {min(self.epoch_losses):.6f} (Epoch {self.epoch_nums[self.epoch_losses.index(min(self.epoch_losses))]})")
        if self.epoch_losses[0] != 0:
            improvement = ((self.epoch_losses[0] - self.epoch_losses[-1])/self.epoch_losses[0]*100)
            print(f"Loss Improvement: {improvement:.2f}%")
        print("="*50 + "\n")

# ==================== DATASET ====================
class ClothingDataset(Dataset):
    """Dataset for clothing images with text captions"""
    
    def __init__(self, image_dir, caption_file, resolution=512):
        self.image_dir = image_dir
        self.resolution = resolution
        
        # Load captions (expecting JSON format: {"image1.jpg": "caption1", ...})
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_paths = list(self.captions.keys())
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)
        caption = "<cloth> "+ self.captions[img_name]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'pixel_values': image,
            'caption': caption
        }

# ==================== TRAINING FUNCTIONS ====================
def collate_fn(batch):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    return {'pixel_values': pixel_values, 'captions': captions}

def train_model(config):
    """Main training function"""
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=config.MIXED_PRECISION
    )
    
    print(f"🚀 Starting training on {accelerator.device}")
    
    # Load pretrained models
    print("📦 Loading pretrained models...")
    tokenizer = CLIPTokenizer.from_pretrained(config.PRETRAINED_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.PRETRAINED_MODEL, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.PRETRAINED_MODEL, subfolder="vae")  # ADD THIS
    unet = UNet2DConditionModel.from_pretrained(config.PRETRAINED_MODEL, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.PRETRAINED_MODEL, subfolder="scheduler")
    
    # Freeze text encoder (we only train UNet)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)  # ADD THIS
    
    # Add LoRA layers to UNet
    print("🔧 Adding LoRA layers...")
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset and dataloader
    print("📁 Loading dataset...")
    dataset = ClothingDataset(config.IMAGE_DIR, config.CAPTION_FILE, config.RESOLUTION)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Optimizer
    if config.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=config.LEARNING_RATE)
            print("✅ Using 8-bit Adam optimizer (saves memory)")
        except ImportError:
            print("⚠️  bitsandbytes not found, using regular AdamW")
            optimizer = torch.optim.AdamW(unet.parameters(), lr=config.LEARNING_RATE)
    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.LEARNING_RATE)
    
    # Prepare with accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    text_encoder = text_encoder.to(accelerator.device)
    vae = vae.to(accelerator.device)
    
    # Initialize metrics tracker
    metrics = MetricsTracker(config.LOGS_DIR)
    
    # Training loop
    print(f"🎯 Training for {config.NUM_EPOCHS} epochs...")
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        unet.train()
        epoch_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # Encode text
                text_inputs = tokenizer(
                    batch['captions'],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(text_inputs)[0]
                
                # Get latents
                pixel_values = batch['pixel_values'].to(accelerator.device)
                with torch.no_grad():  # VAE is frozen
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215  # SD1.5 scaling factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                loss_value = loss.item()
                epoch_loss += loss_value
                num_batches += 1
                global_step += 1
                
                # Log step metrics
                metrics.log_step(loss_value, global_step)
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
        
        avg_loss = epoch_loss / num_batches
        metrics.log_epoch(epoch, avg_loss)
        
        print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-{epoch+1}")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unet.save_pretrained(save_path)
                print(f"💾 Saved checkpoint to {save_path}")
        
        # Generate and save loss curve every epoch
        if accelerator.is_main_process:
            metrics.plot_loss_curve()
    
    # Save final model
    print("🎉 Training completed! Saving final model...")
    final_save_path = os.path.join(config.OUTPUT_DIR, "final_model")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet.save_pretrained(final_save_path)
        print(f"✅ Final model saved to {final_save_path}")
        
        # Save final metrics and plots
        metrics.plot_loss_curve()
        metrics.save_metrics_json()
        metrics.print_summary()

# ==================== INFERENCE FUNCTION ==================== 
def generate_outfit(prompt, lora_path, output_path="generatedoutfit.png"):
    """Generate outfit image from text prompt using trained LoRA"""
    
    print(f"🎨 Generating outfit for: '{prompt}'")
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        Config.PRETRAINED_MODEL,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Load LoRA weights
    pipe.unet = UNet2DConditionModel.from_pretrained(lora_path)
    pipe.unet.to("cuda")
    
    # Generate image
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(output_path)
    print(f"✅ Image saved to {output_path}")
    
    return image

# ==================== MAIN ====================
if __name__ == "__main__":
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Start training
    print("=" * 50)
    print("🎽 OUTFIT GENERATION MODEL TRAINING")
    print("=" * 50)
    train_model(Config)
    
    # Example inference after training
    # generate_outfit(
    #     prompt="a red summer dress with floral patterns",
    #     lora_path=os.path.join(Config.OUTPUT_DIR, "final_model"),
    #     output_path="test_outfit.png"
    # )

    