import os
from PIL import Image, ImageOps, ImageFilter

def preprocess_image(input_path, output_path):
    """
    Applies Gaussian Blur for noise removal and Histogram Equalization for contrast.
    """
    try:
        # Load image
        img = Image.open(input_path).convert('RGB')
        
        # 1. Histogram Equalization (for contrast)
        img = ImageOps.equalize(img)
        
        # 2. Gaussian Blur (for noise removal)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Save image
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    input_dir = 'soil_dataset'
    output_dir = 'preprocessed_soil_dataset'
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print(f"Please ensure '{input_dir}' exists before running this script.")
        return

    print(f"Starting preprocessing from '{input_dir}' to '{output_dir}'...")
    
    processed_count = 0
    error_count = 0
    
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_path = os.path.join(class_dir, filename)
                    # Output path maintaining structure
                    output_path = os.path.join(output_dir, split, class_name, filename)
                    
                    if preprocess_image(input_path, output_path):
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} images...", end='\r')
                    else:
                        error_count += 1

    print(f"\nPreprocessing complete. Processed: {processed_count}, Errors: {error_count}")
    print(f"Preprocessed dataset saved to '{output_dir}'. You can now run train.py.")

if __name__ == '__main__':
    main()
