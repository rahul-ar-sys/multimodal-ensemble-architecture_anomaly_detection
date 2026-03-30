import torch
from src.models.patchcore import PatchCore
import numpy as np

def main():
    print("Testing PatchCore initialization...")
    model = PatchCore(backbone_name="resnet18", coreset_sampling_ratio=0.1)
    
    # Create dummy nominal data loader (batch size 2, 3 channels, 224x224)
    print("Creating dummy nominal images...")
    nominal_images = torch.randn(4, 3, 224, 224)
    dataloader = [nominal_images[i:i+2] for i in range(0, 4, 2)]
    
    print("Fitting model...")
    model.fit(dataloader)
    
    print("Memory bank shape:", model.memory_bank.shape)
    assert model.memory_bank is not None
    
    print("Testing inference...")
    test_images = torch.randn(2, 3, 224, 224)
    anomaly_maps, image_scores = model.predict(test_images)
    
    print("Anomaly Maps shape:", anomaly_maps.shape)
    print("Image Scores shape:", image_scores.shape)
    print("Image Scores:", image_scores)
    
    expected_spatial_dim = 28 # (224 / 8, usually layer 2 or layer 3 stride is 8 or 16. ResNet18 Layer2 is 1/8 so 28x28)
    assert anomaly_maps.shape == (2, expected_spatial_dim, expected_spatial_dim), f"Shape mismatch in anomaly maps. Expected {(2, expected_spatial_dim, expected_spatial_dim)} but got {anomaly_maps.shape}"
    
    print("Verification Passed.")

if __name__ == "__main__":
    main()
