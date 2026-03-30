"""
PatchCore: Base Expert

Captures 'Cold' anomalies (structural deviations from the memory bank).
For the Concat stream, a wider backbone (e.g., WideResNet-101) is recommended
due to increased feature dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
import numpy as np

class PatchCore(nn.Module):
    def __init__(self, backbone_name="resnet18", coreset_sampling_ratio=0.1, n_neighbors=9):
        super(PatchCore, self).__init__()
        self.backbone_name = backbone_name
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.n_neighbors = n_neighbors

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported out of the box.")
        
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # We don't train the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.features = []
        # Hook layer2 and layer3
        self.backbone.layer2.register_forward_hook(self._hook)
        self.backbone.layer3.register_forward_hook(self._hook)
        
        # NearestNeighbor index for memory bank
        self.memory_bank = None
        self.nn_index = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")

    def _hook(self, module, input, output):
        self.features.append(output)

    def _embed(self, images):
        """
        Extract features, apply 3x3 avg pooling, interpolate, and concatenate.
        """
        self.features.clear()
        _ = self.backbone(images)
        
        feat_1 = self.features[0] # layer 2
        feat_2 = self.features[1] # layer 3
        
        # 3x3 average pooling to add neighborhood context
        feat_1 = F.avg_pool2d(feat_1, kernel_size=3, stride=1, padding=1)
        feat_2 = F.avg_pool2d(feat_2, kernel_size=3, stride=1, padding=1)
        
        # Resize feat_2 to feat_1 spatial dimensions for finer resolution
        req_size = feat_1.shape[2:]
        feat_2 = F.interpolate(feat_2, size=req_size, mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        features = torch.cat([feat_1, feat_2], dim=1) # Shape: (B, C1+C2, H, W)
        return features

    def fit(self, dataloader):
        """
        Extract features for all nominal images and perform Coreset subsampling.
        """
        self.backbone.eval()
        all_features = []
        
        with torch.no_grad():
            for images in dataloader:
                if isinstance(images, (list, tuple)):
                    images = images[0] # Assume images are first element if tuple
                images = images.to(self.device)
                features = self._embed(images) # (B, C, H, W)
                
                # Reshape to (B*H*W, C)
                B, C, H, W = features.shape
                features = features.permute(0, 2, 3, 1).reshape(-1, C)
                all_features.append(features.cpu().numpy())
                
        all_features = np.concatenate(all_features, axis=0)
        
        # Coreset Subsampling using Random Projection and K-Center-Greedy approximation
        print(f"Original patch size: {all_features.shape[0]}")
        self.memory_bank = self._get_coreset(all_features)
        print(f"Coreset patch size: {self.memory_bank.shape[0]}")
        
        # Fit Nearest Neighbors
        self.nn_index.fit(self.memory_bank)

    def _get_coreset(self, features):
        """
        K-Center-Greedy approximation for Coreset selection.
        Project to a lower dimension to speed up distance calculations.
        """
        n_samples = features.shape[0]
        coreset_size = int(n_samples * self.coreset_sampling_ratio)
        
        if coreset_size >= n_samples or coreset_size == 0:
            return features
            
        # Random projection for speed (N, C) -> (N, C')
        transformer = SparseRandomProjection(n_components='auto', eps=0.9)
        features_reduced = transformer.fit_transform(features)
        
        # K-Center-Greedy
        coreset_idx = []
        # Random initialization
        init_idx = np.random.randint(0, n_samples)
        coreset_idx.append(init_idx)
        
        # We need a dense array for L2 norm computation
        if hasattr(features_reduced, 'toarray'):
            features_reduced = features_reduced.toarray()

        min_distances = np.linalg.norm(features_reduced - features_reduced[init_idx], axis=1)
        
        # Repeatedly pick the point with the maximum minimum-distance to the chosen set
        for _ in range(1, coreset_size):
            max_idx = np.argmax(min_distances)
            coreset_idx.append(max_idx)
            
            # Update min distances efficiently
            new_distances = np.linalg.norm(features_reduced - features_reduced[max_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
            
        return features[coreset_idx]

    def predict(self, images):
        """
        Inference to get spatial anomaly map and image score.
        """
        self.backbone.eval()
        with torch.no_grad():
            images = images.to(self.device)
            features = self._embed(images) # (B, C, H, W)
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            
            # Query nearest neighbors
            distances, indices = self.nn_index.kneighbors(features, n_neighbors=self.n_neighbors)
            
            # Pixel/Patch anomaly score is the distance to nearest neighbor
            patch_scores = distances[:, 0]
            
            # Reshape back to spatial dimensions
            anomaly_maps = patch_scores.reshape(B, H, W)
            
            # Compute image-level score using PatchCore re-weighting
            image_scores = []
            for i in range(B):
                batch_patch_scores = patch_scores[i*(H*W) : (i+1)*(H*W)]
                batch_distances = distances[i*(H*W) : (i+1)*(H*W)]
                
                max_idx = np.argmax(batch_patch_scores)
                max_score = batch_patch_scores[max_idx]
                
                nn_distances = batch_distances[max_idx]
                weight = 1 - (np.exp(max_score) / np.sum(np.exp(nn_distances)))
                
                image_score = max_score * weight
                image_scores.append(image_score)
                
            return anomaly_maps, np.array(image_scores)
