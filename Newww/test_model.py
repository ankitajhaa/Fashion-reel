"""
Test the trained Pose-Guided GAN model
Generate 5 different poses from a single input image
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from train_pose_guided_gan import Generator
from deepfashion_dataset import get_random_poses, PREDEFINED_POSES
import torchvision.transforms as transforms

class ModelTester:
    """Test the trained model"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        
        # Initialize model
        self.generator = Generator().to(device)
        self.load_model()
        self.generator.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Inverse transform for saving images
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])
    
    def load_model(self):
        """Load trained model weights"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                print(f"✅ Model loaded from {self.model_path}")
            else:
                print(f"❌ Model file {self.model_path} not found. Using untrained model.")
        except Exception as e:
            print(f"❌ Error loading model: {e}. Using untrained model.")
    
    def keypoints_to_heatmap(self, keypoints, height=256, width=256):
        """Convert keypoints to heatmap"""
        heatmap = np.zeros((18, height, width), dtype=np.float32)
        
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Valid keypoint
                # Convert to pixel coordinates
                x_pixel = int(x * width)
                y_pixel = int(y * height)
                
                # Create Gaussian heatmap
                sigma = 2
                for dx in range(-3*sigma, 3*sigma+1):
                    for dy in range(-3*sigma, 3*sigma+1):
                        nx, ny = x_pixel + dx, y_pixel + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist <= 3*sigma:
                                intensity = np.exp(-(dist*dist)/(2*sigma*sigma))
                                heatmap[i, ny, nx] = max(heatmap[i, ny, nx], intensity)
        
        return torch.FloatTensor(heatmap)
    
    def preprocess_image(self, image_path):
        """Preprocess input image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path  # Already a PIL Image
            
            # Apply transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # Return a dummy tensor
            return torch.randn(1, 3, 256, 256).to(self.device)
    
    def generate_poses(self, source_image, num_poses=5):
        """Generate multiple poses from source image"""
        
        # Preprocess source image
        source_tensor = self.preprocess_image(source_image)
        
        # Get target poses
        target_poses = get_random_poses(num_poses)
        
        generated_images = []
        
        with torch.no_grad():
            for pose_info in target_poses:
                # Convert keypoints to heatmap
                pose_heatmap = self.keypoints_to_heatmap(pose_info['keypoints'])
                pose_tensor = pose_heatmap.unsqueeze(0).to(self.device)
                
                # Generate image
                generated_tensor = self.generator(source_tensor, pose_tensor)
                
                # Convert back to PIL Image
                generated_image = self.inverse_transform(generated_tensor.squeeze(0))
                
                generated_images.append({
                    'image': generated_image,
                    'pose_name': pose_info['name'],
                    'keypoints': pose_info['keypoints']
                })
        
        return generated_images
    
    def save_results(self, source_image, generated_images, output_dir='test_results'):
        """Save test results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save source image
        if isinstance(source_image, str):
            source_img = Image.open(source_image)
        else:
            source_img = source_image
        
        source_img.save(os.path.join(output_dir, 'source.jpg'))
        
        # Save generated images
        for i, img_info in enumerate(generated_images):
            filename = f"pose_{i+1}_{img_info['pose_name']}.jpg"
            filepath = os.path.join(output_dir, filename)
            img_info['image'].save(filepath)
            print(f"Saved: {filename}")
        
        print(f"✅ Results saved in {output_dir}/")

def main():
    """Test the model"""
    
    # Find the latest checkpoint
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Using checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoints found. Using untrained model.")
            model_path = None
    else:
        print("No checkpoints directory found. Using untrained model.")
        model_path = None
    
    # Initialize tester
    tester = ModelTester(model_path)
    
    # Test with a sample image from the dataset
    sample_images = []
    img_dir = 'deepfashion_data/img'
    
    # Find a sample image
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(root, file))
                if len(sample_images) >= 1:  # Just get one sample
                    break
        if sample_images:
            break
    
    if sample_images:
        sample_image = sample_images[0]
        print(f"Testing with image: {sample_image}")
        
        # Generate poses
        print("Generating 5 different poses...")
        generated_images = tester.generate_poses(sample_image, num_poses=5)
        
        # Save results
        tester.save_results(sample_image, generated_images)
        
        print("✅ Test completed!")
    else:
        print("❌ No sample images found in dataset")

if __name__ == '__main__':
    main()
