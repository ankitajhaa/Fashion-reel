"""
Pose-Guided Person Image Generation Training Script
Using DeepFashion In-shop Clothes Retrieval Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import inception_v3
import torchvision

import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from collections import OrderedDict
from scipy import linalg
import math

# Model Architecture Components
class PoseEncoder(nn.Module):
    """Encodes pose keypoints into feature maps"""
    def __init__(self, num_keypoints=18, pose_dim=256):
        super(PoseEncoder, self).__init__()
        self.num_keypoints = num_keypoints
        self.pose_dim = pose_dim
        
        # Convert keypoints to heatmaps
        self.conv1 = nn.Conv2d(num_keypoints, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)  # 256x256 -> 128x128
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)  # 128x128 -> 64x64
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)  # 64x64 -> 32x32
        self.conv5 = nn.Conv2d(512, pose_dim, 3, 2, 1)  # 32x32 -> 16x16
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, pose_heatmaps):
        x = F.relu(self.bn1(self.conv1(pose_heatmaps)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class ImageEncoder(nn.Module):
    """Encodes input person image"""
    def __init__(self, input_channels=3, feature_dim=256):
        super(ImageEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        # Encoder layers
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 1, 3)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)  # 256x256 -> 128x128
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)  # 128x128 -> 64x64
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)  # 64x64 -> 32x32
        self.conv5 = nn.Conv2d(512, feature_dim, 3, 2, 1)  # 32x32 -> 16x16
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    """Pose-Guided Generator"""
    def __init__(self, pose_dim=256, image_dim=256, output_channels=3):
        super(Generator, self).__init__()
        
        self.pose_encoder = PoseEncoder(pose_dim=pose_dim)
        self.image_encoder = ImageEncoder(feature_dim=image_dim)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(pose_dim + image_dim, 512, 3, 2, 1, 1)  # 16x16 -> 32x32
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)  # 32x32 -> 64x64
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)  # 64x64 -> 128x128
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)  # 128x128 -> 256x256
        self.deconv5 = nn.Conv2d(64, output_channels, 7, 1, 3)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, source_image, target_pose):
        # Encode source image and target pose
        image_features = self.image_encoder(source_image)
        pose_features = self.pose_encoder(target_pose)
        
        # Concatenate features
        combined_features = torch.cat([image_features, pose_features], dim=1)
        
        # Decode to generate image
        x = F.relu(self.bn1(self.deconv1(combined_features)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        
        return x

class Discriminator(nn.Module):
    """Discriminator for adversarial training"""
    def __init__(self, input_channels=3, pose_channels=18):
        super(Discriminator, self).__init__()
        
        # Image discriminator
        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x

class DeepFashionDataset(Dataset):
    """DeepFashion Dataset Loader"""
    def __init__(self, data_dir, transform=None, pose_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.pose_transform = pose_transform
        
        # Load dataset annotations
        self.image_pairs = self._load_annotations()
        
    def _load_annotations(self):
        """Load image pairs with same clothes but different poses"""
        pairs = []
        
        # For DeepFashion, we need to create pairs of same clothes, different poses
        # This is a simplified version - in practice, you'd load from the actual dataset structure
        
        # Create synthetic pairs for demonstration
        # In real implementation, load from DeepFashion annotations
        for i in range(1000):  # Assuming 1000 pairs
            pairs.append({
                'source_image': f'img_{i}_pose1.jpg',
                'target_image': f'img_{i}_pose2.jpg',
                'source_pose': f'pose_{i}_pose1.json',
                'target_pose': f'pose_{i}_pose2.json'
            })
        
        return pairs
    
    def _keypoints_to_heatmap(self, keypoints, height=256, width=256):
        """Convert keypoints to heatmap"""
        heatmap = np.zeros((18, height, width), dtype=np.float32)
        
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Valid keypoint
                # Create Gaussian heatmap
                x, y = int(x * width), int(y * height)
                if 0 <= x < width and 0 <= y < height:
                    # Simple Gaussian
                    for dx in range(-5, 6):
                        for dy in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist <= 5:
                                    heatmap[i, ny, nx] = max(heatmap[i, ny, nx], np.exp(-dist/2))
        
        return torch.FloatTensor(heatmap)
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load images (in real implementation, load actual images)
        # For now, create dummy data
        source_image = torch.randn(3, 256, 256)
        target_image = torch.randn(3, 256, 256)
        
        # Load poses (in real implementation, load actual keypoints)
        # For now, create dummy keypoints
        source_keypoints = np.random.rand(18, 2)
        target_keypoints = np.random.rand(18, 2)
        
        # Convert to heatmaps
        source_pose = self._keypoints_to_heatmap(source_keypoints)
        target_pose = self._keypoints_to_heatmap(target_keypoints)
        
        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_pose': source_pose,
            'target_pose': target_pose
        }

class PoseGuidedGAN:
    """Main training class"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Training parameters
        self.lambda_adv = 1.0
        self.lambda_l1 = 100.0
        self.lambda_l2 = 10.0
        
        # Evaluation metrics
        self.best_fid = float('inf')
        self.best_ssim = 0.0
        self.training_metrics = {
            'fid_scores': [],
            'ssim_scores': [],
            'l1_scores': [],
            'l2_scores': []
        }
        
    def train_discriminator(self, real_images, fake_images):
        """Train discriminator"""
        self.d_optimizer.zero_grad()
        
        # Real images
        real_pred = self.discriminator(real_images)
        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake images
        fake_pred = self.discriminator(fake_images.detach())
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        
        d_loss = (real_loss + fake_loss) * 0.5
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, source_images, target_poses, target_images):
        """Train generator"""
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        fake_images = self.generator(source_images, target_poses)
        
        # Adversarial loss
        fake_pred = self.discriminator(fake_images)
        adv_loss = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        
        # Reconstruction losses
        l1_loss = self.l1_loss(fake_images, target_images)
        l2_loss = self.l2_loss(fake_images, target_images)
        
        # Total generator loss
        g_loss = (self.lambda_adv * adv_loss + 
                 self.lambda_l1 * l1_loss + 
                 self.lambda_l2 * l2_loss)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), fake_images
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index (SSIM)"""
        # Convert to numpy
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        
        # Calculate SSIM for each image in batch
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            # Convert to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(np.transpose(img1_np[i], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(np.transpose(img2_np[i], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            
            # Calculate SSIM
            ssim = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
            ssim_scores.append(ssim)
        
        return np.mean(ssim_scores)
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate Fréchet Inception Distance (FID) - simplified version"""
        # For now, return a simple metric based on L2 distance
        # In practice, you'd use a pre-trained Inception network
        l2_dist = F.mse_loss(real_images, fake_images)
        return l2_dist.item()
    
    def evaluate_model(self, dataloader, num_samples=100):
        """Evaluate model on validation set"""
        self.generator.eval()
        
        total_l1 = 0
        total_l2 = 0
        total_ssim = 0
        total_fid = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break
                
                source_images = batch['source_image'].to(self.device)
                target_images = batch['target_image'].to(self.device)
                target_poses = batch['target_pose'].to(self.device)
                
                # Generate fake images
                fake_images = self.generator(source_images, target_poses)
                
                # Calculate metrics
                l1_loss = self.l1_loss(fake_images, target_images)
                l2_loss = self.l2_loss(fake_images, target_images)
                ssim_score = self.calculate_ssim(fake_images, target_images)
                fid_score = self.calculate_fid(target_images, fake_images)
                
                total_l1 += l1_loss.item()
                total_l2 += l2_loss.item()
                total_ssim += ssim_score
                total_fid += fid_score
                num_batches += 1
        
        # Average metrics
        avg_l1 = total_l1 / num_batches
        avg_l2 = total_l2 / num_batches
        avg_ssim = total_ssim / num_batches
        avg_fid = total_fid / num_batches
        
        # Update best scores
        if avg_fid < self.best_fid:
            self.best_fid = avg_fid
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
        
        # Store metrics
        self.training_metrics['l1_scores'].append(avg_l1)
        self.training_metrics['l2_scores'].append(avg_l2)
        self.training_metrics['ssim_scores'].append(avg_ssim)
        self.training_metrics['fid_scores'].append(avg_fid)
        
        self.generator.train()
        
        return {
            'l1_loss': avg_l1,
            'l2_loss': avg_l2,
            'ssim': avg_ssim,
            'fid': avg_fid,
            'best_fid': self.best_fid,
            'best_ssim': self.best_ssim
        }
    
    def train(self, dataloader, epochs=100, save_interval=10, eval_interval=5):
        """Main training loop"""
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                source_images = batch['source_image'].to(self.device)
                target_images = batch['target_image'].to(self.device)
                target_poses = batch['target_pose'].to(self.device)
                
                # Train discriminator
                fake_images = self.generator(source_images, target_poses)
                d_loss = self.train_discriminator(target_images, fake_images)
                
                # Train generator
                g_loss, fake_images = self.train_generator(source_images, target_poses, target_images)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}',
                    'D_Loss': f'{d_loss:.4f}'
                })
                
                # Save sample images
                if batch_idx % 100 == 0:
                    self.save_sample_images(source_images, target_poses, fake_images, 
                                          target_images, epoch, batch_idx)
            
            # Evaluate model
            if (epoch + 1) % eval_interval == 0:
                print(f'\nEvaluating model at epoch {epoch+1}...')
                metrics = self.evaluate_model(dataloader)
                print(f'Metrics - L1: {metrics["l1_loss"]:.4f}, L2: {metrics["l2_loss"]:.4f}, '
                      f'SSIM: {metrics["ssim"]:.4f}, FID: {metrics["fid"]:.4f}')
                print(f'Best - SSIM: {metrics["best_ssim"]:.4f}, FID: {metrics["best_fid"]:.4f}')
            
            # Save model checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)
            
            print(f'Epoch {epoch+1}: G_Loss: {epoch_g_loss/len(dataloader):.4f}, '
                  f'D_Loss: {epoch_d_loss/len(dataloader):.4f}')
    
    def save_sample_images(self, source_images, target_poses, fake_images, 
                          target_images, epoch, batch_idx):
        """Save sample images during training"""
        os.makedirs('samples', exist_ok=True)
        
        # Denormalize images (assuming they're in [-1, 1])
        source_images = (source_images + 1) / 2
        fake_images = (fake_images + 1) / 2
        target_images = (target_images + 1) / 2
        
        # Save comparison
        comparison = torch.cat([source_images[:4], fake_images[:4], target_images[:4]], dim=0)
        save_image(comparison, f'samples/epoch_{epoch}_batch_{batch_idx}.png', 
                  nrow=4, normalize=True)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, f'checkpoints/pose_guided_gan_epoch_{epoch}.pth')
        print(f'Checkpoint saved at epoch {epoch}')

def main():
    parser = argparse.ArgumentParser(description='Train Pose-Guided GAN')
    parser.add_argument('--data_dir', type=str, default='./deepfashion_data', 
                       help='Path to DeepFashion dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Using device: {device}')
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = DeepFashionDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = PoseGuidedGAN(device=device)
    
    # Start training
    print('Starting training...')
    model.train(dataloader, epochs=args.epochs)
    
    print('Training completed!')

if __name__ == '__main__':
    main()
