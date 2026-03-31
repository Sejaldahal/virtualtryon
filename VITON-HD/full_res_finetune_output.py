"""
full_res_finetune_output.py

VITON-HD Virtual Try-On Inference Script with Fine-tuned ALIAS Generator
Uses existing dataset structure with preprocessed files.

Usage:
    python vitonhd_tryon_finetuned.py --person 00057_00.jpg --cloth 00094_00.jpg --output result.jpg --dataset_dir "/home/sejal/AI Virtual Tryon/datasets/test" --alias_checkpoint "full_resolution_checkpoints/checkpoint_best.pth"
"""

import os
import argparse
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import json

# Import VITON-HD modules
from networks import SegGenerator, GMM, ALIASGenerator
from utils import save_images


def get_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VITON-HD Virtual Try-On Inference with Fine-tuned ALIAS')
    
    # Required arguments
    parser.add_argument('--person', type=str, required=True, help='Person image filename (e.g., 00057_00.jpg)')
    parser.add_argument('--cloth', type=str, required=True, help='Cloth image filename (e.g., 00094_00.jpg)')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    
    # Dataset paths
    parser.add_argument('--dataset_dir', type=str, default='./test', help='Dataset directory')
    
    # Model paths - IMPORTANT: Now you specify your fine-tuned alias checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Original checkpoint directory')
    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth', help='Segmentation model')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth', help='GMM model')
    parser.add_argument('--alias_checkpoint', type=str, required=True,default='checkpoints/alias_final.pth', 
                       help='Fine-tuned ALIAS checkpoint (e.g., full_resolution_checkpoints/checkpoint_best.pth)')
    
    # Model parameters
    parser.add_argument('--load_height', type=int, default=1024, help='Image height')
    parser.add_argument('--load_width', type=int, default=768, help='Image width')
    
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'alias_state_dict' in state_dict:
            # This is from your fine-tuned checkpoint
            state_dict = state_dict['alias_state_dict']
        
        # Remove 'module.' prefix if present (for DataParallel models)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✓ Successfully loaded checkpoint: {os.path.basename(checkpoint_path)}")
        return True
    except Exception as e:
        print(f"✗ Error loading checkpoint {checkpoint_path}: {e}")
        
        # Try less strict loading
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ Loaded with strict=False (some keys ignored)")
            return True
        except Exception as e2:
            print(f"✗ Failed to load even with strict=False: {e2}")
            return False


def load_finetuned_alias_checkpoint(model, checkpoint_path):
    """Load fine-tuned ALIAS generator from your training checkpoint"""
    print(f"Loading fine-tuned ALIAS from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try different possible keys in your checkpoint
        possible_keys = [
            'alias_state_dict',  # From your save_checkpoint function
            'state_dict',
            'model',
            'alias',
            'generator'
        ]
        
        state_dict = None
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Found state_dict in key: '{key}'")
                break
        
        if state_dict is None:
            # Check if checkpoint is directly the state_dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                print("Checkpoint appears to be direct state_dict")
            else:
                raise KeyError("No valid state_dict found in checkpoint")
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys (these were initialized randomly): {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys (these were ignored): {unexpected_keys}")
        
        print(f"✓ Successfully loaded fine-tuned ALIAS generator")
        return True
        
    except Exception as e:
        print(f"✗ Error loading fine-tuned checkpoint {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


class VITONHDDataset:
    """Dataset handler for single inference using existing VITON-HD structure"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset_dir = opt.dataset_dir
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_parse_agnostic(self, parse, pose_data):
        """Create parse agnostic (from datasets.py)"""
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                      (parse_array == 6).astype(np.float32) +
                      (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        """Create image agnostic (from datasets.py)"""
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                     (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                      (parse_array == 12).astype(np.float32) +
                      (parse_array == 16).astype(np.float32) +
                      (parse_array == 17).astype(np.float32) +
                      (parse_array == 18).astype(np.float32) +
                      (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def load_single_data(self, person_img_name, cloth_img_name):
        """Load single data point using existing dataset structure"""
        print(f"Loading data for person: {person_img_name}, cloth: {cloth_img_name}")
        
        # Load cloth image and mask
        cloth_path = os.path.join(self.dataset_dir, 'cloth', cloth_img_name)
        cloth_mask_path = os.path.join(self.dataset_dir, 'cloth-mask', cloth_img_name)
        
        if not os.path.exists(cloth_path):
            raise FileNotFoundError(f"Cloth image not found: {cloth_path}")
        if not os.path.exists(cloth_mask_path):
            raise FileNotFoundError(f"Cloth mask not found: {cloth_mask_path}")
        
        cloth = Image.open(cloth_path).convert('RGB')
        cloth = transforms.Resize(self.load_width, interpolation=Image.BICUBIC)(cloth)
        cloth_mask = Image.open(cloth_mask_path)
        cloth_mask = transforms.Resize(self.load_width, interpolation=Image.NEAREST)(cloth_mask)
        
        cloth_tensor = self.transform(cloth)
        cloth_mask_array = np.array(cloth_mask)
        cloth_mask_array = (cloth_mask_array >= 128).astype(np.float32)
        cloth_mask_tensor = torch.from_numpy(cloth_mask_array)
        cloth_mask_tensor = cloth_mask_tensor.unsqueeze(0)
        
        # Load pose data
        pose_img_name = person_img_name.replace('.jpg', '_rendered.png')
        pose_img_path = os.path.join(self.dataset_dir, 'openpose-img', pose_img_name)
        pose_json_name = person_img_name.replace('.jpg', '_keypoints.json')
        pose_json_path = os.path.join(self.dataset_dir, 'openpose-json', pose_json_name)
        
        if not os.path.exists(pose_img_path):
            raise FileNotFoundError(f"Pose image not found: {pose_img_path}")
        if not os.path.exists(pose_json_path):
            raise FileNotFoundError(f"Pose JSON not found: {pose_json_path}")
        
        pose_rgb = Image.open(pose_img_path)
        pose_rgb = transforms.Resize(self.load_width, interpolation=Image.BICUBIC)(pose_rgb)
        pose_rgb_tensor = self.transform(pose_rgb)
        
        with open(pose_json_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        
        # Load parsing data
        parse_name = person_img_name.replace('.jpg', '.png')
        parse_path = os.path.join(self.dataset_dir, 'image-parse', parse_name)
        
        if not os.path.exists(parse_path):
            raise FileNotFoundError(f"Parse image not found: {parse_path}")
        
        parse = Image.open(parse_path)
        parse = transforms.Resize(self.load_width, interpolation=Image.NEAREST)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()
        
        # Convert parse agnostic to the proper format
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.opt.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        
        # Load person image
        person_path = os.path.join(self.dataset_dir, 'image', person_img_name)
        if not os.path.exists(person_path):
            raise FileNotFoundError(f"Person image not found: {person_path}")
        
        person_img = Image.open(person_path)
        person_img = transforms.Resize(self.load_width, interpolation=Image.BICUBIC)(person_img)
        img_agnostic = self.get_img_agnostic(person_img, parse, pose_data)
        
        person_tensor = self.transform(person_img)
        img_agnostic_tensor = self.transform(img_agnostic)
        
        return {
            'img_name': person_img_name,
            'cloth_name': cloth_img_name,
            'img': person_tensor.unsqueeze(0),
            'img_agnostic': img_agnostic_tensor.unsqueeze(0),
            'parse_agnostic': new_parse_agnostic_map.unsqueeze(0),
            'pose': pose_rgb_tensor.unsqueeze(0),
            'cloth': cloth_tensor.unsqueeze(0),
            'cloth_mask': cloth_mask_tensor.unsqueeze(0),
        }


class VITONHDTryOn:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 80)
        print("VITON-HD Virtual Try-On with Fine-tuned ALIAS Generator")
        print("=" * 80)
        print(f"Device: {self.device}")
        
        # Set additional options needed for models
        opt.semantic_nc = 13
        opt.grid_size = 5
        opt.norm_G = 'spectralaliasinstance'
        opt.ngf = 64
        opt.num_upsampling_layers = 'most'
        opt.init_type = 'xavier'
        opt.init_variance = 0.02
        
        # Initialize models
        self.seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
        self.gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
        
        # Temporarily change semantic_nc for ALIAS
        original_semantic_nc = opt.semantic_nc
        opt.semantic_nc = 7
        self.alias = ALIASGenerator(opt, input_nc=9)
        opt.semantic_nc = original_semantic_nc
        
        # Load checkpoints
        self.load_models()
        
        # Move to device
        self.seg.to(self.device).eval()
        self.gmm.to(self.device).eval()
        self.alias.to(self.device).eval()
        
        # Gaussian blur for segmentation
        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)
        
        # Dataset handler
        self.dataset = VITONHDDataset(opt)
        
        print("✓ All models loaded successfully!")
        print("=" * 80)
    
    def load_models(self):
        """Load pre-trained model checkpoints"""
        print("\nLoading models...")
        
        # Load original models for seg and gmm
        seg_checkpoint = os.path.join(self.opt.checkpoint_dir, self.opt.seg_checkpoint)
        gmm_checkpoint = os.path.join(self.opt.checkpoint_dir, self.opt.gmm_checkpoint)
        
        if not os.path.exists(seg_checkpoint):
            raise FileNotFoundError(f"Seg checkpoint not found: {seg_checkpoint}")
        if not os.path.exists(gmm_checkpoint):
            raise FileNotFoundError(f"GMM checkpoint not found: {gmm_checkpoint}")
        
        load_checkpoint(self.seg, seg_checkpoint)
        load_checkpoint(self.gmm, gmm_checkpoint)
        
        # Load fine-tuned ALIAS generator
        print(f"\nLoading fine-tuned ALIAS generator from: {self.opt.alias_checkpoint}")
        
        if not os.path.exists(self.opt.alias_checkpoint):
            # Try relative path
            if not os.path.isabs(self.opt.alias_checkpoint):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                alias_path = os.path.join(current_dir, self.opt.alias_checkpoint)
                if os.path.exists(alias_path):
                    self.opt.alias_checkpoint = alias_path
                else:
                    raise FileNotFoundError(f"ALIAS checkpoint not found: {self.opt.alias_checkpoint}")
        
        # First try to load as fine-tuned checkpoint
        if not load_finetuned_alias_checkpoint(self.alias, self.opt.alias_checkpoint):
            print("\n⚠️ Failed to load as fine-tuned checkpoint. Trying original format...")
            
            # Fallback to original checkpoint
            original_alias = os.path.join(self.opt.checkpoint_dir, 'alias_final.pth')
            if os.path.exists(original_alias):
                print(f"Loading original ALIAS from: {original_alias}")
                load_checkpoint(self.alias, original_alias)
            else:
                raise FileNotFoundError(f"Could not load ALIAS from any checkpoint")
    
    def generate_tryon(self, person_img_name, cloth_img_name):
        """Generate virtual try-on result using existing dataset structure"""
        print("\n" + "=" * 80)
        print("Starting virtual try-on generation...")
        print("=" * 80)
        
        data = self.dataset.load_single_data(person_img_name, cloth_img_name)
        
        # Move all tensors to device
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(self.device)
        
        with torch.no_grad():
            # Step 1: Generate segmentation
            print("\nStep 1: Generating segmentation...")
            parse_agnostic_down = F.interpolate(data['parse_agnostic'], size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(data['pose'], size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(data['cloth'] * data['cloth_mask'], size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(data['cloth_mask'], size=(256, 192), mode='bilinear')
            
            # Generate noise for segmentation
            noise = torch.randn(cm_down.size(), device=self.device)
            
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, noise), dim=1)
            parse_pred_down = self.seg(seg_input)
            
            # Upsample and smooth segmentation
            up = nn.Upsample(size=(self.opt.load_height, self.opt.load_width), mode='bilinear')
            parse_pred = self.gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]
            
            parse_old = torch.zeros(parse_pred.size(0), 13, self.opt.load_height, self.opt.load_width, 
                                  dtype=torch.float, device=self.device)
            parse_old.scatter_(1, parse_pred, 1.0)
            
            # Convert to 7-class format
            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]]
            }
            
            parse = torch.zeros(parse_pred.size(0), 7, self.opt.load_height, self.opt.load_width, 
                              dtype=torch.float, device=self.device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]
            
            # Step 2: Warp cloth using GMM
            print("Step 2: Warping cloth...")
            agnostic_gmm = F.interpolate(data['img_agnostic'], size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(data['pose'], size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(data['cloth'], size=(256, 192), mode='nearest')
            
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            _, warped_grid = self.gmm(gmm_input, c_gmm)
            
            warped_c = F.grid_sample(data['cloth'], warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(data['cloth_mask'], warped_grid, padding_mode='border')
            
            # Step 3: Generate final try-on result with fine-tuned ALIAS
            print("Step 3: Generating final result with fine-tuned ALIAS...")
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask
            
            # ALIAS generator input
            alias_input = torch.cat((data['img_agnostic'], data['pose'], warped_c), dim=1)
            
            # Generate final output
            output = self.alias(alias_input, parse, parse_div, misalign_mask)
            
            print("✓ Try-on generation completed!")
            return output


def save_image_tensor(tensor, output_path):
    """Save tensor as image"""
    # Convert tensor to PIL Image
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first batch
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    tensor = tensor.cpu().detach()
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW to HWC
    tensor_np = tensor.numpy()
    tensor_np = (tensor_np * 255).astype(np.uint8)
    
    # Save as image
    img = Image.fromarray(tensor_np)
    img.save(output_path)
    print(f"Saved result to: {output_path}")


def main():
    opt = get_opt()
    
    print(f"\nConfiguration:")
    print(f"  Dataset directory: {opt.dataset_dir}")
    print(f"  Original checkpoints: {opt.checkpoint_dir}")
    print(f"  Fine-tuned ALIAS: {opt.alias_checkpoint}")
    print(f"  Person image: {opt.person}")
    print(f"  Cloth image: {opt.cloth}")
    print(f"  Output: {opt.output}")
    
    # Validate inputs
    if not os.path.exists(opt.dataset_dir):
        print(f"\n❌ Error: Dataset directory not found: {opt.dataset_dir}")
        return
    
    # Check if required subdirectories exist
    required_dirs = ['image', 'cloth', 'cloth-mask', 'image-parse', 'openpose-img', 'openpose-json']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(opt.dataset_dir, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Error: Missing required directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return
    
    print(f"\n✓ All required directories found")
    
    # Check if specific files exist
    person_path = os.path.join(opt.dataset_dir, 'image', opt.person)
    cloth_path = os.path.join(opt.dataset_dir, 'cloth', opt.cloth)
    
    if not os.path.exists(person_path):
        print(f"\n❌ Error: Person image not found: {person_path}")
        return
    else:
        print(f"✓ Found person image: {person_path}")
    
    if not os.path.exists(cloth_path):
        print(f"\n❌ Error: Cloth image not found: {cloth_path}")
        return
    else:
        print(f"✓ Found cloth image: {cloth_path}")
    
    # Check original checkpoint directory
    if not os.path.exists(opt.checkpoint_dir):
        print(f"\n❌ Error: Original checkpoint directory not found: {opt.checkpoint_dir}")
        print("Please download the pre-trained checkpoints for seg and gmm.")
        return
    
    # Check fine-tuned ALIAS checkpoint
    if not os.path.exists(opt.alias_checkpoint):
        # Try relative path
        if not os.path.isabs(opt.alias_checkpoint):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alias_path = os.path.join(current_dir, opt.alias_checkpoint)
            if os.path.exists(alias_path):
                opt.alias_checkpoint = alias_path
            else:
                print(f"\n❌ Error: Fine-tuned ALIAS checkpoint not found: {opt.alias_checkpoint}")
                print("Please provide the correct path to your fine-tuned checkpoint.")
                return
    
    try:
        # Initialize try-on system
        viton = VITONHDTryOn(opt)
        
        # Generate try-on result
        result = viton.generate_tryon(opt.person, opt.cloth)
        
        # Save result
        output_dir = os.path.dirname(opt.output) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        person_base = os.path.splitext(opt.person)[0]
        cloth_base = os.path.splitext(opt.cloth)[0]
        output_name = f"{person_base}_{cloth_base}_finetuned.jpg"
        output_path = os.path.join(output_dir, output_name)
        
        # Save the result
        save_image_tensor(result, output_path)
        
        print(f"\n" + "=" * 80)
        print("🎉 SUCCESS!")
        print(f"✓ Fine-tuned try-on result saved to: {output_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during try-on generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


# Usage Examples:
# ============================================================
# 1. Using your fine-tuned checkpoint:
#    python full_res_finetune_output.py \
#      --person sample.jpg \
#      --cloth sample.jpg \
#      --output ./results/ \
#      --dataset_dir "/home/sejal/AI Virtual Tryon/preprocessed" \
#      --alias_checkpoint "checkpoints/alias_final.pth"

# 2. Using a different fine-tuned checkpoint:
#    python full_res_finetune_output.py \
#      --person 00013_00.jpg \
#      --cloth 00071_00.jpg \
#      --output ./results/ \
#      --alias_checkpoint "full_resolution_checkpoints/checkpoint_epoch_1.pth" 
#
# 3. Using relative paths:
#    python full_res_finetune_output.py \
#      --person 00057_00.jpg \
#      --cloth 00094_00.jpg \
#      --output ./results/result.jpg \
#      --dataset_dir "./datasets/test" \
#      --alias_checkpoint "./checkpoints/alias_final.pth"
# ============================================================
#python full_res_finetune_output.py      --person sample.jpg      --cloth sample.jpg      --output ./results/      --dataset_dir "/home/sejal/AI Virtual Tryon/preprocessed"      --alias_checkpoint "checkpoints/alias_final.pth"