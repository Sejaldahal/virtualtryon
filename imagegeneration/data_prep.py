"""
COMPLETE TXT TO JSON CONVERTER WITH ENHANCED CAPTIONS
Converts folder of .txt caption files to single .json file
With comprehensive clothing detail enhancement
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# ==================== CAPTION ENHANCEMENT ====================
def enhance_caption(caption):
    """
    Intelligently enhance BLIP captions with meaningful clothing details
    Adds: colors, sleeve styles, neck styles, patterns, fits, and more
    """
    
    # Clean up common BLIP artifacts
    caption = caption.replace("arafed", "").replace("araffed", "")
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # Don't enhance if caption is already detailed (>100 chars)
    if len(caption) > 100:
        return caption
    
    enhanced = caption
    caption_lower = caption.lower()
    
    # === COLOR DETECTION ===
    colors_found = []
    color_keywords = {
        'black': 'black', 'white': 'white', 'red': 'red', 'blue': 'blue', 
        'green': 'green', 'yellow': 'yellow', 'pink': 'pink', 'purple': 'purple',
        'orange': 'orange', 'brown': 'brown', 'gray': 'gray', 'grey': 'grey',
        'navy': 'navy blue', 'beige': 'beige', 'cream': 'cream', 'tan': 'tan',
        'burgundy': 'burgundy', 'maroon': 'maroon', 'olive': 'olive green',
        'teal': 'teal', 'turquoise': 'turquoise', 'gold': 'gold', 'silver': 'silver',
        'khaki': 'khaki', 'ivory': 'ivory', 'charcoal': 'charcoal', 'mint': 'mint',
        'lavender': 'lavender', 'coral': 'coral', 'peach': 'peach'
    }
    
    for keyword, color_name in color_keywords.items():
        if keyword in caption_lower and color_name not in colors_found:
            colors_found.append(color_name)
    
    # === GARMENT DETECTION & ENHANCEMENT ===
    
    # T-SHIRTS & TOPS
    if any(word in caption_lower for word in ['tshirt', 't-shirt', 't shirt', 'shirt', 'top', 'blouse']):
        details = []
        
        # Necklines
        if 'v-neck' in caption_lower or 'v neck' in caption_lower:
            details.append("v-neck")
        elif 'scoop' in caption_lower:
            details.append("scoop neck")
        elif 'boat' in caption_lower:
            details.append("boat neck")
        elif 'square' in caption_lower and 'neck' in caption_lower:
            details.append("square neckline")
        elif 'off shoulder' in caption_lower:
            details.append("off-shoulder")
        elif 'halter' in caption_lower:
            details.append("halter neck")
        elif 'turtle' in caption_lower or 'high neck' in caption_lower:
            details.append("turtleneck")
        elif 'collar' in caption_lower or 'polo' in caption_lower:
            details.append("collared")
        elif 'crew' in caption_lower:
            details.append("crew neck")
        else:
            details.append("round neck")
        
        # Sleeves
        if 'sleeveless' in caption_lower or 'tank' in caption_lower:
            details.append("sleeveless")
        elif 'cap sleeve' in caption_lower:
            details.append("cap sleeves")
        elif 'short sleeve' in caption_lower or ('short' in caption_lower and 'sleeve' in caption_lower):
            details.append("short sleeves")
        elif '3/4 sleeve' in caption_lower:
            details.append("3/4 sleeves")
        elif 'long sleeve' in caption_lower or ('long' in caption_lower and 'sleeve' in caption_lower):
            details.append("long sleeves")
        elif 'bell sleeve' in caption_lower:
            details.append("bell sleeves")
        elif 'puff sleeve' in caption_lower:
            details.append("puff sleeves")
        else:
            details.append("short sleeves")
        
        # Patterns
        if 'stripe' in caption_lower:
            details.append("striped pattern")
        elif 'polka dot' in caption_lower or 'dot' in caption_lower:
            details.append("polka dot")
        elif 'floral' in caption_lower or 'flower' in caption_lower:
            details.append("floral print")
        elif 'graphic' in caption_lower or 'logo' in caption_lower:
            details.append("graphic print")
        elif 'text' in caption_lower:
            details.append("text design")
        elif 'animal' in caption_lower or 'leopard' in caption_lower:
            details.append("animal print")
        elif 'tie dye' in caption_lower:
            details.append("tie-dye")
        elif 'check' in caption_lower or 'plaid' in caption_lower:
            details.append("checkered")
        elif 'plain' in caption_lower or 'solid' in caption_lower:
            details.append("solid color")
        
        # Fit
        if 'crop' in caption_lower:
            details.append("cropped")
        if 'fitted' in caption_lower or 'tight' in caption_lower:
            details.append("fitted")
        elif 'loose' in caption_lower or 'oversized' in caption_lower:
            details.append("oversized")
        
        if details:
            enhanced += ", " + ", ".join(details)
    
    # PANTS & JEANS
    elif any(word in caption_lower for word in ['jeans', 'pants', 'trousers', 'denim', 'leggings']):
        details = []
        
        # Type
        if 'jeans' in caption_lower or 'denim' in caption_lower:
            details.append("denim")
        elif 'leather' in caption_lower:
            details.append("leather pants")
        elif 'cargo' in caption_lower:
            details.append("cargo style")
        elif 'legging' in caption_lower:
            details.append("stretch leggings")
        
        # Fit
        if 'skinny' in caption_lower:
            details.append("skinny fit")
        elif 'slim' in caption_lower:
            details.append("slim fit")
        elif 'straight' in caption_lower:
            details.append("straight leg")
        elif 'wide' in caption_lower:
            details.append("wide leg")
        elif 'bootcut' in caption_lower:
            details.append("bootcut")
        elif 'flare' in caption_lower:
            details.append("flared")
        
        # Rise
        if 'high waist' in caption_lower or 'high-waist' in caption_lower:
            details.append("high-waisted")
        elif 'low rise' in caption_lower:
            details.append("low-rise")
        
        # Wash
        if 'dark wash' in caption_lower:
            details.append("dark wash")
        elif 'light wash' in caption_lower:
            details.append("light wash")
        elif 'faded' in caption_lower:
            details.append("faded")
        
        # Details
        if 'ripped' in caption_lower or 'distressed' in caption_lower:
            details.append("distressed")
        
        if details:
            enhanced += ", " + ", ".join(details)
    
    # DRESSES
    elif 'dress' in caption_lower:
        details = []
        
        # Length
        if 'mini' in caption_lower:
            details.append("mini length")
        elif 'midi' in caption_lower:
            details.append("midi length")
        elif 'maxi' in caption_lower:
            details.append("maxi length")
        elif 'knee' in caption_lower:
            details.append("knee-length")
        
        # Sleeves
        if 'sleeveless' in caption_lower:
            details.append("sleeveless")
        elif 'short sleeve' in caption_lower:
            details.append("short sleeves")
        elif 'long sleeve' in caption_lower:
            details.append("long sleeves")
        
        # Neckline
        if 'v-neck' in caption_lower:
            details.append("v-neck")
        elif 'square' in caption_lower:
            details.append("square neckline")
        elif 'halter' in caption_lower:
            details.append("halter neck")
        elif 'off shoulder' in caption_lower:
            details.append("off-shoulder")
        
        # Pattern
        if 'floral' in caption_lower:
            details.append("floral pattern")
        elif 'stripe' in caption_lower:
            details.append("striped")
        elif 'polka dot' in caption_lower:
            details.append("polka dot")
        elif 'lace' in caption_lower:
            details.append("lace details")
        
        # Style
        if 'wrap' in caption_lower:
            details.append("wrap style")
        elif 'a-line' in caption_lower:
            details.append("A-line")
        elif 'bodycon' in caption_lower or 'fitted' in caption_lower:
            details.append("bodycon")
        
        if details:
            enhanced += ", " + ", ".join(details)
    
    # JACKETS & OUTERWEAR
    elif any(word in caption_lower for word in ['jacket', 'coat', 'hoodie', 'cardigan', 'blazer', 'sweater']):
        details = []
        
        # Material
        if 'leather' in caption_lower:
            details.append("leather")
        elif 'denim' in caption_lower:
            details.append("denim jacket")
        elif 'bomber' in caption_lower:
            details.append("bomber style")
        elif 'puffer' in caption_lower:
            details.append("puffer jacket")
        elif 'fleece' in caption_lower:
            details.append("fleece")
        elif 'knit' in caption_lower:
            details.append("knit fabric")
        
        # Closure
        if 'zip' in caption_lower or 'zipper' in caption_lower:
            details.append("zip closure")
        elif 'button' in caption_lower:
            details.append("button closure")
        
        # Features
        if 'hood' in caption_lower or 'hoodie' in caption_lower:
            details.append("with hood")
        if 'pocket' in caption_lower:
            details.append("with pockets")
        
        # Fit
        if 'oversized' in caption_lower:
            details.append("oversized")
        elif 'fitted' in caption_lower:
            details.append("fitted")
        
        if details:
            enhanced += ", " + ", ".join(details)
    
    # SKIRTS
    elif 'skirt' in caption_lower:
        details = []
        
        # Length
        if 'mini' in caption_lower:
            details.append("mini length")
        elif 'midi' in caption_lower:
            details.append("midi length")
        elif 'maxi' in caption_lower:
            details.append("maxi length")
        
        # Style
        if 'pleated' in caption_lower:
            details.append("pleated")
        elif 'a-line' in caption_lower:
            details.append("A-line")
        elif 'pencil' in caption_lower:
            details.append("pencil skirt")
        
        # Pattern
        if 'floral' in caption_lower:
            details.append("floral")
        elif 'stripe' in caption_lower:
            details.append("striped")
        elif 'plaid' in caption_lower:
            details.append("plaid")
        
        if 'high waist' in caption_lower:
            details.append("high-waisted")
        
        if details:
            enhanced += ", " + ", ".join(details)
    
    # Add color info
    if colors_found and len(enhanced) < 150:
        color_str = " and ".join(colors_found)
        if color_str.lower() not in enhanced.lower():
            enhanced += f", {color_str} color"
    
    # Add fabric if missing and short
    if len(enhanced) < 60:
        if 'tshirt' in caption_lower or 't-shirt' in caption_lower:
            if 'cotton' not in enhanced.lower():
                enhanced += ", cotton blend"
    
    return enhanced

# ==================== MAIN CONVERSION FUNCTION ====================
def convert_txt_to_json(txt_folder, image_folder, output_json, enhance=True):
    """
    Convert folder of .txt caption files to single JSON file
    
    Args:
        txt_folder: Path to folder containing .txt caption files
        image_folder: Path to folder containing images (to verify they exist)
        output_json: Output JSON file path
        enhance: Whether to enhance captions with additional details
    """
    
    print("=" * 70)
    print("🔄 CONVERTING TXT CAPTIONS TO JSON")
    print("=" * 70)
    print(f"📁 TXT Folder: {txt_folder}")
    print(f"🖼️  Image Folder: {image_folder}")
    print(f"📄 Output JSON: {output_json}")
    print(f"✨ Enhancement: {'ENABLED' if enhance else 'DISABLED'}")
    print("=" * 70)
    
    # Get all .txt files
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    print(f"\n📝 Found {len(txt_files)} .txt caption files")
    
    # Get all image files for verification
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions}
    print(f"🖼️  Found {len(image_files)} images in image folder")
    
    # Process each txt file
    captions = {}
    enhanced_count = 0
    missing_images = []
    empty_captions = []
    
    print("\n🔄 Processing caption files...")
    
    for txt_file in tqdm(txt_files, desc="Converting"):
        # Get base name without .txt extension
        base_name = os.path.splitext(txt_file)[0]
        
        # Read caption
        txt_path = os.path.join(txt_folder, txt_file)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Skip empty captions
            if not caption:
                empty_captions.append(txt_file)
                continue
            
            # Enhance caption if requested
            original_length = len(caption)
            if enhance:
                caption = enhance_caption(caption)
                if len(caption) > original_length:
                    enhanced_count += 1
            
            # Find corresponding image file
            if base_name in image_files:
                image_filename = image_files[base_name]
                captions[image_filename] = caption
            else:
                missing_images.append(base_name)
                
        except Exception as e:
            print(f"\n⚠️  Error reading {txt_file}: {e}")
    
    # Save to JSON
    print(f"\n💾 Saving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully converted: {len(captions)} image-caption pairs")
    
    if enhance:
        print(f"✨ Enhanced captions: {enhanced_count} ({enhanced_count/len(captions)*100:.1f}%)")
    
    if empty_captions:
        print(f"\n⚠️  Empty captions: {len(empty_captions)} files")
        print(f"   (Skipped during conversion)")
    
    if missing_images:
        print(f"\n⚠️  Missing images: {len(missing_images)} files")
        print(f"   (Caption exists but no matching image found)")
        
        # Save list
        missing_file = "missing_images.txt"
        with open(missing_file, 'w') as f:
            f.write('\n'.join(missing_images))
        print(f"   List saved to: {missing_file}")
    
    print(f"\n📊 Final dataset: {len(captions)} image-caption pairs")
    print(f"📝 Output: {output_json}")
    print("=" * 70)
    
    return captions

# ==================== PREVIEW FUNCTION ====================
def preview_conversions(json_file, num_samples=10):
    """Preview sample conversions"""
    
    import random
    
    with open(json_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"\n🎲 PREVIEW: Random {num_samples} samples")
    print("=" * 70)
    
    samples = random.sample(list(captions.items()), min(num_samples, len(captions)))
    
    for i, (img_file, caption) in enumerate(samples, 1):
        print(f"\n{i}. 🖼️  {img_file}")
        print(f"   📝 {caption}")
        print(f"   [{len(caption)} chars]")
    
    print("\n" + "=" * 70)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    TXT_FOLDER = "/home/sejal/AI Virtual Tryon/datasets/train/captionofcloths"
    IMAGE_FOLDER = "/home/sejal/AI Virtual Tryon/datasets/train/cloth"
    OUTPUT_JSON = "/home/sejal/AI Virtual Tryon/datasets/train/captions_enhanced.json"
    ENHANCE_CAPTIONS = True  # Set to False for no enhancement
    
    print("\n🎽 TXT TO JSON CONVERTER - VIRTUAL TRY-ON DATASET")
    
    # Convert
    captions = convert_txt_to_json(
        txt_folder=TXT_FOLDER,
        image_folder=IMAGE_FOLDER,
        output_json=OUTPUT_JSON,
        enhance=ENHANCE_CAPTIONS
    )
    
    # Preview
    preview_conversions(OUTPUT_JSON, num_samples=10)
    
    print("\n✅ Ready for training!")
    print(f"👉 Use '{OUTPUT_JSON}' in your training script\n")