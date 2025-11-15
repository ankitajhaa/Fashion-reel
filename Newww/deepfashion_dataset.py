"""
DeepFashion Dataset Loader for Pose-Guided Person Image Generation
This module handles loading and preprocessing the DeepFashion In-shop Clothes Retrieval Dataset
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
import random

class DeepFashionDataset(Dataset):
    """
    DeepFashion In-shop Clothes Retrieval Dataset
    Creates pairs of same clothes, different poses for pose transfer training
    """
    
    def __init__(self, data_dir, mode='train', transform=None, pose_transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.pose_transform = pose_transform
        
        # Paths
        self.img_dir = os.path.join(data_dir, 'img')
        self.pose_dir = os.path.join(data_dir, 'pose')
        self.anno_file = os.path.join(data_dir, 'Anno', 'list_bbox_inshop.txt')
        
        # Load dataset
        self.image_pairs = self._load_annotations()
        
        print(f"Loaded {len(self.image_pairs)} image pairs for {mode} mode")
    
    def _load_annotations(self):
        """Load image pairs with same clothes but different poses"""
        pairs = []
        
        # Group images by clothes ID (same clothes, different poses)
        clothes_groups = defaultdict(list)
        
        # Get all image files recursively
        all_images = self._get_all_image_files()
        
        # Group by clothes ID (folder name)
        for img_path in all_images:
            # Extract clothes ID from path (folder name)
            rel_path = os.path.relpath(img_path, self.img_dir)
            clothes_id = os.path.dirname(rel_path)  # This gives us the folder path as clothes ID
            
            # Check if pose file exists
            pose_path = os.path.join(self.pose_dir, rel_path.replace('.jpg', '_keypoints.json'))
            
            if os.path.exists(pose_path):
                clothes_groups[clothes_id].append({
                    'img_path': img_path,
                    'pose_path': pose_path,
                    'img_name': os.path.basename(img_path)
                })
        
        # Create pairs from same clothes group
        for clothes_id, images in clothes_groups.items():
            if len(images) >= 2:  # Need at least 2 poses for same clothes
                # Create all possible pairs
                for i in range(len(images)):
                    for j in range(len(images)):
                        if i != j:  # Different poses
                            pairs.append({
                                'source': images[i],
                                'target': images[j],
                                'clothes_id': clothes_id
                            })
        
        # If no real data, create synthetic pairs for demonstration
        if not pairs:
            print("No real data found, creating synthetic pairs for demonstration...")
            pairs = self._create_synthetic_pairs()
        
        return pairs
    
    def _get_all_image_files(self):
        """Get all image files recursively"""
        image_files = []
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        return image_files
    
    def _extract_clothes_id(self, img_name):
        """Extract clothes ID from image name"""
        # DeepFashion naming convention: clothes_id_pose_id.jpg
        # Extract the clothes ID part
        parts = img_name.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:-1])  # Everything except last part (pose_id)
        return img_name.split('.')[0]
    
    def _create_synthetic_pairs(self):
        """Create synthetic pairs for demonstration when real data is not available"""
        pairs = []
        
        # Create 1000 synthetic pairs
        for i in range(1000):
            pairs.append({
                'source': {
                    'img_path': f'synthetic_img_{i}_pose1.jpg',
                    'pose_path': f'synthetic_pose_{i}_pose1.json',
                    'img_name': f'synthetic_img_{i}_pose1.jpg'
                },
                'target': {
                    'img_path': f'synthetic_img_{i}_pose2.jpg',
                    'pose_path': f'synthetic_pose_{i}_pose2.json',
                    'img_name': f'synthetic_img_{i}_pose2.jpg'
                },
                'clothes_id': f'clothes_{i}'
            })
        
        return pairs
    
    def _load_keypoints(self, pose_path):
        """Load pose keypoints from JSON file"""
        try:
            if os.path.exists(pose_path):
                with open(pose_path, 'r') as f:
                    data = json.load(f)
                    if 'people' in data and len(data['people']) > 0:
                        keypoints = data['people'][0]['pose_keypoints_2d']
                        # Reshape to (18, 3) - 18 keypoints, each with (x, y, confidence)
                        keypoints = np.array(keypoints).reshape(-1, 3)
                        return keypoints[:, :2]  # Only x, y coordinates
            
            # If no real keypoints, generate synthetic ones
            return self._generate_synthetic_keypoints()
            
        except Exception as e:
            print(f"Error loading keypoints from {pose_path}: {e}")
            return self._generate_synthetic_keypoints()
    
    def _generate_synthetic_keypoints(self):
        """Generate synthetic keypoints for demonstration"""
        # COCO format: 18 keypoints
        keypoints = np.zeros((18, 2))
        
        # Generate realistic pose keypoints
        # Head
        keypoints[0] = [0.5, 0.1]  # nose
        keypoints[1] = [0.45, 0.08]  # left_eye
        keypoints[2] = [0.55, 0.08]  # right_eye
        keypoints[3] = [0.42, 0.12]  # left_ear
        keypoints[4] = [0.58, 0.12]  # right_ear
        
        # Torso
        keypoints[5] = [0.4, 0.25]  # left_shoulder
        keypoints[6] = [0.6, 0.25]  # right_shoulder
        keypoints[7] = [0.4, 0.4]   # left_elbow
        keypoints[8] = [0.6, 0.4]   # right_elbow
        keypoints[9] = [0.4, 0.55]  # left_wrist
        keypoints[10] = [0.6, 0.55] # right_wrist
        
        # Lower body
        keypoints[11] = [0.45, 0.6] # left_hip
        keypoints[12] = [0.55, 0.6] # right_hip
        keypoints[13] = [0.45, 0.8] # left_knee
        keypoints[14] = [0.55, 0.8] # right_knee
        keypoints[15] = [0.45, 0.95] # left_ankle
        keypoints[16] = [0.55, 0.95] # right_ankle
        
        # Add some random variation
        noise = np.random.normal(0, 0.02, keypoints.shape)
        keypoints += noise
        
        # Ensure keypoints are within [0, 1] range
        keypoints = np.clip(keypoints, 0, 1)
        
        return keypoints
    
    def _keypoints_to_heatmap(self, keypoints, height=256, width=256):
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
    
    def _load_image(self, img_path):
        """Load and preprocess image"""
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                # Create synthetic image for demonstration
                image = Image.new('RGB', (256, 256), color=(128, 128, 128))
                # Add some random patterns to make it look like a person
                import numpy as np
                img_array = np.array(image)
                # Add some random colored regions
                img_array[50:200, 50:200] = np.random.randint(0, 255, (150, 150, 3))
                image = Image.fromarray(img_array)
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                image = transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            return torch.randn(3, 256, 256)
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load source image and pose
        source_image = self._load_image(pair['source']['img_path'])
        source_keypoints = self._load_keypoints(pair['source']['pose_path'])
        source_pose = self._keypoints_to_heatmap(source_keypoints)
        
        # Load target image and pose
        target_image = self._load_image(pair['target']['img_path'])
        target_keypoints = self._load_keypoints(pair['target']['pose_path'])
        target_pose = self._keypoints_to_heatmap(target_keypoints)
        
        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_pose': source_pose,
            'target_pose': target_pose,
            'source_keypoints': torch.FloatTensor(source_keypoints),
            'target_keypoints': torch.FloatTensor(target_keypoints),
            'clothes_id': pair['clothes_id']
        }

def create_dataloader(data_dir, batch_size=8, num_workers=4, mode='train'):
    """Create DataLoader for DeepFashion dataset"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = DeepFashionDataset(data_dir, mode=mode, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode == 'train'), 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

# Predefined poses for inference
PREDEFINED_POSES = {
    'standing_front': np.array([
        [0.5, 0.1],   # nose
        [0.45, 0.08], [0.55, 0.08],  # eyes
        [0.42, 0.12], [0.58, 0.12],  # ears
        [0.4, 0.25], [0.6, 0.25],    # shoulders
        [0.4, 0.4], [0.6, 0.4],      # elbows
        [0.4, 0.55], [0.6, 0.55],    # wrists
        [0.45, 0.6], [0.55, 0.6],    # hips
        [0.45, 0.8], [0.55, 0.8],    # knees
        [0.45, 0.95], [0.55, 0.95]   # ankles
    ]),
    
    'hand_on_hip': np.array([
        [0.5, 0.1],   # nose
        [0.45, 0.08], [0.55, 0.08],  # eyes
        [0.42, 0.12], [0.58, 0.12],  # ears
        [0.4, 0.25], [0.6, 0.25],    # shoulders
        [0.35, 0.35], [0.6, 0.4],    # elbows (left hand on hip)
        [0.3, 0.3], [0.6, 0.55],     # wrists
        [0.45, 0.6], [0.55, 0.6],    # hips
        [0.45, 0.8], [0.55, 0.8],    # knees
        [0.45, 0.95], [0.55, 0.95]   # ankles
    ]),
    
    'walking_pose': np.array([
        [0.5, 0.1],   # nose
        [0.45, 0.08], [0.55, 0.08],  # eyes
        [0.42, 0.12], [0.58, 0.12],  # ears
        [0.4, 0.25], [0.6, 0.25],    # shoulders
        [0.4, 0.4], [0.6, 0.4],      # elbows
        [0.4, 0.55], [0.6, 0.55],    # wrists
        [0.4, 0.6], [0.6, 0.6],      # hips (slight shift)
        [0.4, 0.8], [0.6, 0.8],      # knees
        [0.35, 0.95], [0.65, 0.95]   # ankles (walking stance)
    ]),
    
    'side_pose': np.array([
        [0.4, 0.1],   # nose (side view)
        [0.38, 0.08], [0.42, 0.08],  # eyes
        [0.35, 0.12], [0.45, 0.12],  # ears
        [0.3, 0.25], [0.5, 0.25],    # shoulders
        [0.25, 0.4], [0.5, 0.4],     # elbows
        [0.2, 0.55], [0.5, 0.55],    # wrists
        [0.35, 0.6], [0.55, 0.6],    # hips
        [0.35, 0.8], [0.55, 0.8],    # knees
        [0.35, 0.95], [0.55, 0.95]   # ankles
    ]),
    
    'sitting_pose': np.array([
        [0.5, 0.15],  # nose (higher due to sitting)
        [0.45, 0.13], [0.55, 0.13],  # eyes
        [0.42, 0.17], [0.58, 0.17],  # ears
        [0.4, 0.3], [0.6, 0.3],      # shoulders
        [0.4, 0.45], [0.6, 0.45],    # elbows
        [0.4, 0.6], [0.6, 0.6],      # wrists
        [0.45, 0.7], [0.55, 0.7],    # hips (sitting position)
        [0.45, 0.85], [0.55, 0.85],  # knees (bent)
        [0.45, 0.95], [0.55, 0.95]   # ankles
    ])
}

def get_random_poses(num_poses=5):
    """Get random poses for inference"""
    pose_names = list(PREDEFINED_POSES.keys())
    selected_poses = random.sample(pose_names, min(num_poses, len(pose_names)))
    
    poses = []
    for pose_name in selected_poses:
        keypoints = PREDEFINED_POSES[pose_name]
        # Add some random variation
        noise = np.random.normal(0, 0.02, keypoints.shape)
        keypoints += noise
        keypoints = np.clip(keypoints, 0, 1)
        poses.append({
            'name': pose_name,
            'keypoints': keypoints
        })
    
    return poses

if __name__ == '__main__':
    # Test the dataset
    data_dir = './deepfashion_data'
    dataloader = create_dataloader(data_dir, batch_size=4)
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Test loading a batch
    for batch in dataloader:
        print(f"Source image shape: {batch['source_image'].shape}")
        print(f"Target image shape: {batch['target_image'].shape}")
        print(f"Source pose shape: {batch['source_pose'].shape}")
        print(f"Target pose shape: {batch['target_pose'].shape}")
        break
