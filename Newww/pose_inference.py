"""
Pose-Guided Person Image Generation Inference
This module handles generating new poses from a single input image
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from deepfashion_dataset import get_random_poses, PREDEFINED_POSES
from train_pose_guided_gan import Generator, PoseEncoder, ImageEncoder
import torchvision.transforms as transforms
import argparse

class PoseInference:
    """Pose inference class for generating new poses"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        
        # Initialize model
        self.generator = Generator().to(device)
        
        # Load trained model
        self.load_model()
        
        # Set to evaluation mode
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
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Model file {self.model_path} not found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}. Using untrained model.")
    
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
    
    def generate_poses(self, source_image, num_poses=5, pose_names=None):
        """Generate multiple poses from source image"""
        
        # Preprocess source image
        source_tensor = self.preprocess_image(source_image)
        
        # Get target poses
        if pose_names is None:
            target_poses = get_random_poses(num_poses)
        else:
            target_poses = []
            for name in pose_names:
                if name in PREDEFINED_POSES:
                    target_poses.append({
                        'name': name,
                        'keypoints': PREDEFINED_POSES[name]
                    })
        
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
    
    def save_generated_images(self, generated_images, output_dir='generated_poses'):
        """Save generated images to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, img_info in enumerate(generated_images):
            filename = f"pose_{i+1}_{img_info['pose_name']}.jpg"
            filepath = os.path.join(output_dir, filename)
            img_info['image'].save(filepath)
            saved_paths.append(filepath)
        
        return saved_paths

class PoseDetector:
    """Simple pose detector for input images (if needed)"""
    
    def __init__(self):
        self.num_keypoints = 18
    
    def detect_pose(self, image_path):
        """Detect pose in input image"""
        # This is a simplified version
        # In practice, you'd use OpenPose, MediaPipe, or similar
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._generate_default_pose()
            
            # For demonstration, return a default pose
            # In real implementation, use actual pose detection
            return self._generate_default_pose()
            
        except Exception as e:
            print(f"Error detecting pose: {e}")
            return self._generate_default_pose()
    
    def _generate_default_pose(self):
        """Generate a default standing pose"""
        return np.array([
            [0.5, 0.1],   # nose
            [0.45, 0.08], [0.55, 0.08],  # eyes
            [0.42, 0.12], [0.58, 0.12],  # ears
            [0.4, 0.25], [0.6, 0.25],    # shoulders
            [0.4, 0.4], [0.6, 0.4],      # elbows
            [0.4, 0.55], [0.6, 0.55],    # wrists
            [0.45, 0.6], [0.55, 0.6],    # hips
            [0.45, 0.8], [0.55, 0.8],    # knees
            [0.45, 0.95], [0.55, 0.95]   # ankles
        ])

def create_reel_from_images(image_paths, output_path='reel.mp4', duration_per_image=4, fps=30):
    """Create a video reel from generated images"""
    try:
        from moviepy.editor import ImageSequenceClip, concatenate_videoclips
        
        # Create clips from images
        clips = []
        for img_path in image_paths:
            clip = ImageSequenceClip([img_path], durations=[duration_per_image])
            clips.append(clip)
        
        # Concatenate clips
        final_clip = concatenate_videoclips(clips)
        
        # Write video
        final_clip.write_videofile(output_path, fps=fps, codec='libx264')
        
        print(f"Reel created: {output_path}")
        return output_path
        
    except ImportError:
        print("MoviePy not available. Install with: pip install moviepy")
        return None
    except Exception as e:
        print(f"Error creating reel: {e}")
        return None

def main():
    """Test the inference pipeline"""
    parser = argparse.ArgumentParser(description='Pose-Guided Inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained checkpoint (.pth). If not provided, pick latest in checkpoints/.')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image. If not provided, use test_results/source.jpg or create dummy.')
    parser.add_argument('--num_poses', type=int, default=5, help='Number of poses to generate')
    parser.add_argument('--out_dir', type=str, default='generated_poses', help='Output directory')
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model_path
    if model_path is None:
        ckpt_dir = 'checkpoints'
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            if ckpts:
                latest = max(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                model_path = os.path.join(ckpt_dir, latest)
                print(f"Using latest checkpoint: {latest}")
            else:
                print('No checkpoints found. Proceeding with untrained model.')
                model_path = ''
        else:
            print('No checkpoints directory found. Proceeding with untrained model.')
            model_path = ''

    inference = PoseInference(model_path)

    # Resolve input image
    sample_image = args.image
    if sample_image is None:
        # Prefer user's source.jpg if present
        candidate = os.path.join('test_results', 'source.jpg')
        if os.path.exists(candidate):
            sample_image = candidate
        else:
            # Fallback: create a dummy
            sample_image = 'sample_input.jpg'
            if not os.path.exists(sample_image):
                print(f"Sample image {sample_image} not found. Creating a dummy image.")
                dummy_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
                dummy_image.save(sample_image)

    # Generate poses
    print("Generating poses...")
    generated_images = inference.generate_poses(sample_image, num_poses=args.num_poses)

    # Save images
    saved_paths = inference.save_generated_images(generated_images, output_dir=args.out_dir)
    print(f"Generated {len(saved_paths)} images -> {args.out_dir}")

    # Create reel
    reel_path = create_reel_from_images(saved_paths)
    if reel_path:
        print(f"Reel created: {reel_path}")

if __name__ == '__main__':
    main()
