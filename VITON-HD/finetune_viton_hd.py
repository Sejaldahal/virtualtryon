"""
enhanced_viton_hd_finetune.py

Enhanced fine-tuning script for VITON-HD virtual try-on model.

TASK CLARIFICATION:
- Input: Person wearing cloth A + Target cloth B
- Output: Person wearing cloth B (virtual try-on result)
- Training: Use paired data where person wears both cloth A and B
"""

import os
import argparse
import time
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchgeometry as tgm
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the original VITON-HD modules
from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, save_images


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for realistic image synthesis"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_sigmoid=False):
        super().__init__()
        
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, nf_mult * ndf,
                     kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(nf_mult * ndf),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(nf_mult * ndf, 1, kernel_size=kw, stride=1, padding=padw)]
        
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    """GAN loss with different modes"""
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features"""
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(pretrained=True).features.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.vgg_layers = vgg[:16]
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except Exception as e:
            print(f"Warning: Could not load VGG16: {e}")
            self.vgg_layers = None

    def forward(self, x, y):
        if self.vgg_layers is None:
            return torch.tensor(0.0, device=x.device)
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)
        
        return F.l1_loss(x_features, y_features)


def safe_load_state_dict(model, checkpoint_path, strict=True):
    """Safely load state dict with various format handling"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                model.load_state_dict(checkpoint, strict=strict)
                return True
            
            for key in ['state_dict', 'model_state_dict', 'model', 'generator', 'alias', 'gmm', 'seg']:
                if key in checkpoint:
                    try:
                        model.load_state_dict(checkpoint[key], strict=strict)
                        return True
                    except:
                        try:
                            new_sd = {k.replace('module.', ''): v for k, v in checkpoint[key].items()}
                            model.load_state_dict(new_sd, strict=strict)
                            return True
                        except:
                            continue
            
            model_sd = model.state_dict()
            matching_sd = {}
            for name, param in model_sd.items():
                if name in checkpoint and checkpoint[name].shape == param.shape:
                    matching_sd[name] = checkpoint[name]
                else:
                    for ckpt_key, ckpt_param in checkpoint.items():
                        if name in ckpt_key and ckpt_param.shape == param.shape:
                            matching_sd[name] = ckpt_param
                            break
            
            if len(matching_sd) > 0:
                model.load_state_dict(matching_sd, strict=False)
                print(f"Loaded {len(matching_sd)}/{len(model_sd)} parameters")
                return True
        else:
            model.load_state_dict(checkpoint, strict=strict)
            return True
            
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
    
    return False


def get_opt():
    """Get training options (with flexible resize options)."""
    parser = argparse.ArgumentParser(description='VITON-HD Virtual Try-On Fine-tuning')

    # Training parameters - Memory optimized defaults
    parser.add_argument('--name', type=str, default='viton_hd_enhanced', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (reduced default for memory)')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Number of data workers (reduced for stability)')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency (steps)')
    parser.add_argument('--display_freq', type=int, default=500, help='Display frequency (steps)')

    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='Dataset directory')
    parser.add_argument('--dataset_mode', type=str, default='train', help='Dataset split')
    parser.add_argument('--dataset_list', type=str, default='train_pairs.txt', help='Pair list file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='Pre-trained checkpoints')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_enhanced/', help='Save directory')
    parser.add_argument('--result_dir', type=str, default='./results_enhanced/', help='Results directory')
    parser.add_argument('--log_dir', type=str, default='./logs_enhanced/', help='Logs directory')

    # Model parameters (must match original VITON-HD)
    parser.add_argument('--load_height', type=int, default=1024, help='Original image height (repo default)')
    parser.add_argument('--load_width', type=int, default=768, help='Original image width (repo default)')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Shuffle data')

    # Flexible resizing to reduce memory use:
    parser.add_argument('--resize_factor', type=float, default=1.0,
                        help='Scale factor applied to checkpoint/original resolution before building models. '
                             'e.g. 0.5 for half resolution. If >0, overrides target_height/width if set to 0.')
    parser.add_argument('--target_height', type=int, default=0,
                        help='If >0, explicit target height to use for training (overrides resize_factor).')
    parser.add_argument('--target_width', type=int, default=0,
                        help='If >0, explicit target width to use for training (overrides resize_factor).')

    # Checkpoint paths
    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth', help='Seg checkpoint name')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth', help='GMM checkpoint name')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth', help='ALIAS checkpoint name')

    # Architecture parameters
    parser.add_argument('--semantic_nc', type=int, default=13, help='Semantic classes')
    parser.add_argument('--init_type', type=str, default='xavier', help='Init type')
    parser.add_argument('--init_variance', type=float, default=0.02, help='Init variance')
    parser.add_argument('--grid_size', type=int, default=5, help='GMM grid size')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='Generator norm')
    parser.add_argument('--ngf', type=int, default=64, help='Generator filters')
    parser.add_argument('--num_upsampling_layers', type=str, default='most', help='Upsampling layers')

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=10.0, help='L1 loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')

    # Training options
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision (default: True)')
    parser.add_argument('--resume', type=str, default='', help='Resume checkpoint path')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['vanilla', 'lsgan', 'wgangp'])

    # Memory optimization
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Accumulate gradients')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')

    return parser.parse_args()



def setup_models(opt, device):
    """Initialize all models"""
    print("Initializing models...")
    
    # Segmentation Generator (predicts parsing for target cloth)
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    
    # Geometric Matching Module (warps target cloth to fit person)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    
    # ALIAS Generator (generates final try-on result)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13
    
    # Discriminator (distinguishes real from generated images)
    discriminator = PatchGANDiscriminator(input_nc=3, ndf=64)
    
    # Move to device
    models = [seg, gmm, alias, discriminator]
    for model in models:
        model.to(device)
    
    return seg, gmm, alias, discriminator


def load_checkpoints(seg, gmm, alias, opt, device):
    """Load pre-trained checkpoints - REQUIRED for fine-tuning"""
    print("Loading pre-trained VITON-HD checkpoints...")
    print("Note: All three checkpoints (seg_final.pth, gmm_final.pth, alias_final.pth) are required!")
    
    checkpoints = [
        (seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint), "SegGenerator"),
        (gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint), "GMM"),
        (alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint), "ALIASGenerator")
    ]
    
    all_loaded = True
    for model, path, name in checkpoints:
        if os.path.exists(path):
            if safe_load_state_dict(model, path, strict=False):
                print(f"✓ Loaded pre-trained {name} from {os.path.basename(path)}")
            else:
                print(f"✗ Failed to load {name} from {path}")
                all_loaded = False
        else:
            print(f"✗ CRITICAL: {name} checkpoint not found: {path}")
            print(f"    Please ensure {os.path.basename(path)} exists in {opt.checkpoint_dir}")
            all_loaded = False
    
    if not all_loaded:
        raise FileNotFoundError(
            "Missing required pre-trained checkpoints! Fine-tuning requires:\n"
            f"  - {opt.checkpoint_dir}/{opt.seg_checkpoint}\n"
            f"  - {opt.checkpoint_dir}/{opt.gmm_checkpoint}\n"
            f"  - {opt.checkpoint_dir}/{opt.alias_checkpoint}\n"
            "Please download the pre-trained VITON-HD models first."
        )
    
    print("All pre-trained checkpoints loaded successfully - ready for fine-tuning!")
    return seg, gmm, alias


def prepare_virtual_tryon_data(batch, seg, gmm, opt, device):
    """
    Prepare data for virtual try-on training.
    
    VIRTUAL TRY-ON TASK:
    - Input: Person + Target Cloth B
    - Output: Person wearing Cloth B
    - Ground Truth: Real image of person wearing cloth (for training supervision)
    
    During training:
    - We have paired data: person wearing cloth A AND person wearing cloth B
    - We use person+cloth_B as input, person_wearing_cloth_B as target
    """
    with torch.no_grad():
        # Input components
        img_agnostic = batch['img_agnostic'].to(device)      # Person with original cloth removed
        parse_agnostic = batch['parse_agnostic'].to(device)   # Segmentation with original cloth removed
        pose = batch['pose'].to(device)                       # Person's pose keypoints
        
        # Target cloth to try on
        target_cloth = batch['cloth']['unpaired'].to(device)     # Target cloth B
        target_cloth_mask = batch['cloth_mask']['unpaired'].to(device)  # Target cloth B mask
        
        # Ground truth (person wearing the target cloth - for training only)
        ground_truth = batch['img'].to(device)  # Real image: person wearing cloth
        
        # Create blur filter for segmentation smoothing
        up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)
        
        # Step 1: Generate segmentation for target cloth
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        cloth_masked_down = F.interpolate(target_cloth * target_cloth_mask, size=(256, 192), mode='bilinear')
        cloth_mask_down = F.interpolate(target_cloth_mask, size=(256, 192), mode='bilinear')
        
        # Generate noise for segmentation (as in original code)
        noise = torch.randn(cloth_mask_down.size(), device=device) * 0.5 + 0.5
        seg_input = torch.cat((cloth_mask_down, cloth_masked_down, parse_agnostic_down, pose_down, noise), dim=1)
        
        # Predict segmentation
        parse_pred_down = seg(seg_input)
        parse_pred = gauss(up(parse_pred_down))
        parse_pred = parse_pred.argmax(dim=1)[:, None]
        
        # Convert 13-channel segmentation to 7-channel (as per original code)
        parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, 
                               dtype=torch.float, device=device)
        parse_old.scatter_(1, parse_pred, 1.0)
        
        # Label mapping (from original code)
        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]], 
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        
        parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, 
                           dtype=torch.float, device=device)
        for j in range(len(labels)):
            for label in labels[j][1]:
                parse[:, j] += parse_old[:, label]
        
        # Step 2: Warp target cloth to fit person using GMM
        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')  # Upper cloth region
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        cloth_gmm = F.interpolate(target_cloth, size=(256, 192), mode='nearest')
        
        # GMM input: person info + target cloth region
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
        
        # Get warping grid and apply to target cloth
        _, warped_grid = gmm(gmm_input, cloth_gmm)
        warped_cloth = F.grid_sample(target_cloth, warped_grid, padding_mode='border', align_corners=False)
        warped_cloth_mask = F.grid_sample(target_cloth_mask, warped_grid, padding_mode='border', align_corners=False)
        
        # Step 3: Prepare ALIAS Generator inputs
        # Handle misalignment between predicted cloth region and actual warped cloth
        misalign_mask = parse[:, 2:3] - warped_cloth_mask
        misalign_mask[misalign_mask < 0.0] = 0.0
        
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask  # Remove misaligned regions
        
        # Final input to ALIAS generator
        alias_input = torch.cat((img_agnostic, pose, warped_cloth), dim=1)
        
        return {
            'alias_input': alias_input,      # Input: person + pose + warped target cloth
            'parse': parse,                  # 7-channel segmentation  
            'parse_div': parse_div,         # 8-channel segmentation with misalignment
            'misalign_mask': misalign_mask, # Misalignment mask
            'ground_truth': ground_truth,   # Target: person wearing the cloth (supervision)
            'target_cloth': target_cloth,   # Original target cloth (for visualization)
            'warped_cloth': warped_cloth    # Warped target cloth (for visualization)
        }


def save_checkpoint(alias, discriminator, optimizer_g, optimizer_d, epoch, opt, losses, best=False):
    """Save training checkpoint"""
    os.makedirs(opt.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'alias_state_dict': alias.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'losses': losses,
        'opt': vars(opt)
    }
    
    filename = f'checkpoint_best.pth' if best else f'checkpoint_epoch_{epoch}.pth'
    filepath = os.path.join(opt.save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filename}")


def plot_losses(losses, save_path):
    """Plot and save training losses"""
    epochs = range(1, len(losses['g_loss']) + 1)
    
    plt.figure(figsize=(18, 6))
    
    # Generator losses subplot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses['g_loss'], 'b-', linewidth=2, label='Total G Loss')
    plt.plot(epochs, losses['g_adv'], 'r-', linewidth=1, label='G Adversarial')
    plt.plot(epochs, losses['g_l1'], 'g-', linewidth=1, label='G L1')
    plt.plot(epochs, losses['g_perceptual'], 'm-', linewidth=1, label='G Perceptual')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Discriminator losses subplot  
    plt.subplot(1, 3, 2)
    plt.plot(epochs, losses['d_loss'], 'orange', linewidth=2, label='Total D Loss')
    plt.plot(epochs, losses['d_real'], 'cyan', linewidth=1, label='D Real')
    plt.plot(epochs, losses['d_fake'], 'purple', linewidth=1, label='D Fake')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Generator vs Discriminator
    plt.subplot(1, 3, 3)
    plt.plot(epochs, losses['g_loss'], 'b-', linewidth=2, label='Generator')
    plt.plot(epochs, losses['d_loss'], 'r-', linewidth=2, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {save_path}")


def save_sample_images(real_images, fake_images, target_cloths, warped_cloths, save_path, epoch, step):
    """Save sample images for visualization"""
    # Limit to first 4 samples for display
    real_imgs = real_images[:4]
    fake_imgs = fake_images[:4] 
    cloths = target_cloths[:4]
    warped = warped_cloths[:4]
    
    # Create grid: [real, fake, target_cloth, warped_cloth] x batch_size
    all_imgs = torch.cat([real_imgs, fake_imgs, cloths, warped], dim=0)
    
    grid = vutils.make_grid(all_imgs, nrow=4, normalize=True, scale_each=True, padding=2)
    vutils.save_image(grid, save_path)


def main():
    opt = get_opt()
    print("=" * 80)
    print("VITON-HD FINE-TUNING (Virtual Try-On)")
    print("=" * 80)
    print("IMPORTANT: This script performs FINE-TUNING on pre-trained models")
    print("Required checkpoints in --checkpoint_dir:")
    print(f"  - {opt.seg_checkpoint}")
    print(f"  - {opt.gmm_checkpoint}")  
    print(f"  - {opt.alias_checkpoint}")
    print("=" * 80)
    print(f"Task: Generate person wearing target cloth")
    print(f"Input: Person + Target Cloth → Output: Person wearing Target Cloth")
    print("=" * 80)
    
    # Create directories
    for dir_path in [opt.save_dir, opt.result_dir, opt.log_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup models
    seg, gmm, alias, discriminator = setup_models(opt, device)
    
    # Load pre-trained weights
    seg, gmm, alias = load_checkpoints(seg, gmm, alias, opt, device)
    
    # Freeze pre-trained models (only fine-tune ALIAS generator)
    seg.eval()
    gmm.eval()
    for param in seg.parameters():
        param.requires_grad = False
    for param in gmm.parameters():
        param.requires_grad = False
    
    # Set training modes
    alias.train()
    discriminator.train()
    
    # Setup data loader
    print("Loading dataset...")
    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader.data_loader)}")
    print(f"Workers: {opt.workers}, Batch size: {opt.batch_size}")
    
    # Optimizers and loss functions
    optimizer_g = torch.optim.Adam(alias.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    
    criterionGAN = GANLoss(opt.gan_mode).to(device)
    criterionL1 = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    # Mixed precision scalers
    scaler_g = GradScaler(enabled=opt.amp)
    scaler_d = GradScaler(enabled=opt.amp)
    
    # Loss tracking
    losses = {
        'g_loss': [], 'g_adv': [], 'g_l1': [], 'g_perceptual': [],
        'd_loss': [], 'd_real': [], 'd_fake': []
    }
    
    # Training state
    start_epoch = 1
    best_loss = float('inf')
    
    # Resume fine-tuning if specified (different from loading pre-trained checkpoints)
    if opt.resume and os.path.exists(opt.resume):
        print(f"\nResuming FINE-TUNING from: {opt.resume}")
        print("Note: This is different from loading pre-trained checkpoints above")
        checkpoint = torch.load(opt.resume, map_location='cpu')
        
        # Load fine-tuning states (not the original pre-trained weights)
        alias.load_state_dict(checkpoint['alias_state_dict'])  # Fine-tuned ALIAS weights
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])  # Discriminator state
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint.get('losses', losses)
        print(f"Resumed fine-tuning from epoch {start_epoch}")
    else:
        print(f"\nStarting fresh fine-tuning (epoch 1)")
        print("Pre-trained models loaded above will be used as starting point")
    
    # Debug: Test data loading before training
    print("Testing data loader...")
    try:
        test_batch = next(iter(train_loader.data_loader))
        print(f"✓ Successfully loaded test batch with {len(test_batch['img_name'])} samples")
        print(f"✓ Sample image names: {test_batch['img_name'][:3]}")
        print(f"✓ Sample cloth names: {[test_batch['c_name']['unpaired'][i] for i in range(min(3, len(test_batch['c_name']['unpaired'])))]}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        print("Check your dataset structure and train_pairs.txt")
        raise e
    
    print(f"Fine-tuning from epoch {start_epoch} to {opt.epochs}")
    print("=" * 80)
    
    # Main training loop
    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_losses = {key: 0.0 for key in losses.keys()}
        
        # Progress bar for current epoch
        pbar = tqdm(train_loader.data_loader, 
                   desc=f'Epoch {epoch}/{opt.epochs}',
                   unit='batch',
                   dynamic_ncols=True)
        
        for step, batch in enumerate(pbar, 1):
            # Prepare virtual try-on data
            tryon_data = prepare_virtual_tryon_data(batch, seg, gmm, opt, device)
            
            alias_input = tryon_data['alias_input']
            parse = tryon_data['parse']
            parse_div = tryon_data['parse_div']
            misalign_mask = tryon_data['misalign_mask']
            ground_truth = tryon_data['ground_truth']  # Real person wearing cloth
            
            # ==========================================
            # Update Discriminator
            # ==========================================
            
            with autocast(enabled=opt.amp):
                # Generate fake try-on result
                fake_tryon = alias(alias_input, parse, parse_div, misalign_mask)
                
                # Discriminator on real images (ground truth)
                pred_real = discriminator(ground_truth)
                loss_d_real = criterionGAN(pred_real, True)
                
                # Discriminator on fake images (generated try-on)
                pred_fake = discriminator(fake_tryon.detach())
                loss_d_fake = criterionGAN(pred_fake, False)
                
                # Combined discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5 / opt.gradient_accumulation_steps
            
            # Backward pass for discriminator
            scaler_d.scale(loss_d).backward()
            
            if step % opt.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), opt.max_grad_norm)
                
                scaler_d.step(optimizer_d)
                scaler_d.update()
                optimizer_d.zero_grad()
            
            # ==========================================
            # Update Generator (ALIAS)
            # ==========================================
            
            with autocast(enabled=opt.amp):
                # Adversarial loss - fool the discriminator
                pred_fake = discriminator(fake_tryon)
                loss_g_adv = criterionGAN(pred_fake, True) * opt.lambda_adv
                
                # L1 reconstruction loss - match ground truth
                loss_g_l1 = criterionL1(fake_tryon, ground_truth) * opt.lambda_l1
                
                # Perceptual loss - preserve high-level features
                loss_g_perceptual = perceptual_loss(fake_tryon, ground_truth) * opt.lambda_perceptual
                
                # Combined generator loss
                loss_g = (loss_g_adv + loss_g_l1 + loss_g_perceptual) / opt.gradient_accumulation_steps
            
            # Backward pass for generator
            scaler_g.scale(loss_g).backward()
            
            if step % opt.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(alias.parameters(), opt.max_grad_norm)
                
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()
            
            # ==========================================
            # Memory cleanup
            # ==========================================
            if step % 100 == 0:
                torch.cuda.empty_cache()
            
            # ==========================================
            # Update loss tracking
            # ==========================================
            epoch_losses['g_loss'] += loss_g.item()
            epoch_losses['g_adv'] += loss_g_adv.item()
            epoch_losses['g_l1'] += loss_g_l1.item()
            epoch_losses['g_perceptual'] += loss_g_perceptual.item()
            epoch_losses['d_loss'] += loss_d.item()
            epoch_losses['d_real'] += loss_d_real.item()
            epoch_losses['d_fake'] += loss_d_fake.item()
            
            # Update progress bar with current losses
            pbar.set_postfix({
                'G': f'{loss_g.item():.3f}',
                'D': f'{loss_d.item():.3f}',
                'L1': f'{loss_g_l1.item():.3f}',
                'Adv': f'{loss_g_adv.item():.3f}'
            })
            
            # Save sample images periodically
            if step % opt.display_freq == 0:
                with torch.no_grad():
                    sample_path = f'{opt.result_dir}/epoch_{epoch:03d}_step_{step:05d}.png'
                    save_sample_images(
                        ground_truth, fake_tryon, 
                        tryon_data['target_cloth'], tryon_data['warped_cloth'],
                        sample_path, epoch, step
                    )
            
            # Print detailed loss information
            if step % opt.print_freq == 0:
                print(f"\n[Epoch {epoch}/{opt.epochs}] [Step {step}/{len(train_loader.data_loader)}]")
                print(f"  Generator - Total: {loss_g.item():.4f} | "
                      f"L1: {loss_g_l1.item():.4f} | "
                      f"Adv: {loss_g_adv.item():.4f} | "
                      f"Perc: {loss_g_perceptual.item():.4f}")
                print(f"  Discriminator - Total: {loss_d.item():.4f} | "
                      f"Real: {loss_d_real.item():.4f} | "
                      f"Fake: {loss_d_fake.item():.4f}")
        
        # ==========================================
        # End of Epoch Processing
        # ==========================================
        
        # Calculate average losses for the epoch
        num_batches = len(train_loader.data_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch}/{opt.epochs}] Summary:")
        print(f"  Generator Loss: {epoch_losses['g_loss']:.4f}")
        print(f"    - Adversarial: {epoch_losses['g_adv']:.4f}")
        print(f"    - L1: {epoch_losses['g_l1']:.4f}")  
        print(f"    - Perceptual: {epoch_losses['g_perceptual']:.4f}")
        print(f"  Discriminator Loss: {epoch_losses['d_loss']:.4f}")
        print(f"    - Real: {epoch_losses['d_real']:.4f}")
        print(f"    - Fake: {epoch_losses['d_fake']:.4f}")
        
        # Check for training stability
        d_real_acc = (epoch_losses['d_real'] < 0.5)
        d_fake_acc = (epoch_losses['d_fake'] < 0.5)
        if d_real_acc and d_fake_acc:
            print(f"  ✓ Discriminator is learning well")
        elif not d_real_acc:
            print(f"  ⚠ Discriminator struggling with real images")
        elif not d_fake_acc:
            print(f"  ⚠ Generator may be too strong")
        
        print(f"{'='*80}")
        
        # Save checkpoints
        if epoch % opt.save_freq == 0 or epoch == opt.epochs:
            save_checkpoint(alias, discriminator, optimizer_g, optimizer_d, epoch, opt, losses)
        
        # Save best model based on generator loss
        if epoch_losses['g_loss'] < best_loss:
            best_loss = epoch_losses['g_loss']
            save_checkpoint(alias, discriminator, optimizer_g, optimizer_d, epoch, opt, losses, best=True)
            print(f"🏆 New best model saved! Generator Loss: {best_loss:.4f}")
        
        # Update loss plots
        plot_losses(losses, os.path.join(opt.log_dir, 'training_losses.png'))
        
        # Save loss history as JSON
        with open(os.path.join(opt.log_dir, 'loss_history.json'), 'w') as f:
            json.dump(losses, f, indent=2)
        
        # Create a final sample at end of epoch
        with torch.no_grad():
            final_sample_path = f'{opt.result_dir}/epoch_{epoch:03d}_final.png'
            save_sample_images(
                ground_truth[:8], fake_tryon[:8], 
                tryon_data['target_cloth'][:8], tryon_data['warped_cloth'][:8],
                final_sample_path, epoch, 'final'
            )
    
    # ==========================================
    # Training Completed
    # ==========================================
    
    print("\n" + "="*80)
    print("🎉 FINE-TUNING COMPLETED!")
    print("="*80)
    print("Final Results:")
    print(f"  - Best Generator Loss: {best_loss:.4f}")
    print(f"  - Total Fine-tuning Epochs: {opt.epochs}")
    print(f"  - Training Samples: {len(train_dataset)}")
    print("="*80)
    print("Files saved to:")
    print(f"  - Fine-tuned checkpoints: {opt.save_dir}")
    print(f"  - Loss plots: {os.path.join(opt.log_dir, 'training_losses.png')}")
    print(f"  - Loss data: {os.path.join(opt.log_dir, 'loss_history.json')}")
    print(f"  - Sample images: {opt.result_dir}")
    print("="*80)
    print("NOTE: Original pre-trained models in --checkpoint_dir remain unchanged")
    print("Fine-tuned models are saved separately in --save_dir")


if __name__ == '__main__':
    main()
    
    
    