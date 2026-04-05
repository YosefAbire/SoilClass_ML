import os
import random
import shutil

def split_dataset(data_dir='soil_dataset', val_split=0.15, test_split=0.15):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Ensure train directory exists
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist.")
        return
        
    # Get all classes
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found classes: {classes}")
    
    total_moved = 0
    
    for cls in classes:
        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)
        
        # Ensure validation and test class directories exist
        os.makedirs(cls_val_dir, exist_ok=True)
        os.makedirs(cls_test_dir, exist_ok=True)
        
        # Get all images for this class
        images = [f for f in os.listdir(cls_train_dir) if os.path.isfile(os.path.join(cls_train_dir, f))]
        
        # Shuffle images
        random.shuffle(images)
        
        total_images = len(images)
        num_val = int(total_images * val_split)
        num_test = int(total_images * test_split)
        
        val_images = images[:num_val]
        test_images = images[num_val:num_val+num_test]
        
        # Move images to validation
        for img in val_images:
            src = os.path.join(cls_train_dir, img)
            dst = os.path.join(cls_val_dir, img)
            shutil.move(src, dst)
            total_moved += 1
            
        # Move images to test
        for img in test_images:
            src = os.path.join(cls_train_dir, img)
            dst = os.path.join(cls_test_dir, img)
            shutil.move(src, dst)
            total_moved += 1
            
        print(f"Class '{cls}': {total_images} total -> moved {num_val} to validation, {num_test} to test. left {total_images - num_val - num_test} in train.")

    print(f"Successfully moved {total_moved} images total.")

if __name__ == "__main__":
    split_dataset()
