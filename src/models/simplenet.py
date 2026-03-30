#standard
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter

# --- 1. DATASET DEFINITION ---
class BottleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filters only image files from the directory
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                            if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self): 
        return len(self.image_files)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: 
            img = self.transform(img)
        return img

# --- 2. ENHANCED FEATURE EXTRACTOR ---
class IndustryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # WideResNet-101 provides deeper feature refinement for subtle defects
        backbone = models.wide_resnet101_2(weights='IMAGENET1K_V2')
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])
        self.layer2 = list(backbone.children())[5] # 512 channels
        self.layer3 = list(backbone.children())[6] # 1024 channels
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            # Spatial Alignment via bilinear interpolation
            x3_up = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=True)
            return torch.cat([x2, x3_up], dim=1) # 1536 fused channels

# --- 3. THE SIMPLENET MODULE ---
class SimpleNetPro(nn.Module):
    def __init__(self, in_channels=1536):
        super().__init__()
        # Adapter removes domain bias from pre-trained ImageNet features
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 512, 1) 
        )
        # Discriminator scores features based on learned decision boundary
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        adapted = self.adapter(x)
        b, c, h, w = adapted.shape
        flat = adapted.permute(0, 2, 3, 1).reshape(-1, c)
        return self.discriminator(flat).view(b, 1, h, w)

# --- 4. LOSS FUNCTION ---
def stable_anomaly_loss(pos_scores, neg_scores, margin=0.5):
    # Truncated L2 Loss prevents the model from "flipping" its logic
    loss_pos = torch.mean(torch.clamp(margin - pos_scores, min=0))
    loss_neg = torch.mean(torch.clamp(margin + neg_scores, min=0))
    return loss_pos + loss_neg

# --- 5. EXECUTION PIPELINE ---
def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    extractor = IndustryBackbone().to(device)
    model = SimpleNetPro().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Industry standard 512x512 resolution for minute crack detection
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BottleDataset("dataset/train/good", transform=data_transform), 
                              batch_size=4, shuffle=True)

    print("Training Optimized SimpleNet...")
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            features = extractor(batch)
            pos_scores = model(features)
            
            # Synthetic anomaly scale increased to 0.05 for clear boundary learning
            noise = torch.randn_like(features) * 0.01
            neg_scores = model(features + noise)
            
            loss = stable_anomaly_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0: print(f"Epoch {epoch} complete")

    print("\nEvaluating and Visualizing Results...")
    y_true, y_scores = [], []
    model.eval()
    
    test_paths = [(1, "dataset/test/structural_damage"), (0, "dataset/test/good")]
    for label, folder in test_paths:
        if not os.path.exists(folder): continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn))
            img_t = data_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Smooth score maps to reduce pixel-level noise
                score_map = torch.sigmoid(model(extractor(img_t))).cpu().numpy()[0,0]
                score_map = cv2.resize(score_map, (512, 512))
                score_map = gaussian_filter(score_map, sigma=4)
                
                y_scores.append(np.max(score_map))
                y_true.append(label)

    # Final ROC metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'SimpleNet (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.savefig("final_simplenet_roc.png")
    
    print(f"Final AUROC: {auroc:.4f}. Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()



#IR
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter

# --- 1. DATASET DEFINITION ---
class BottleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filters only image files from the directory
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                            if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self): 
        return len(self.image_files)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: 
            img = self.transform(img)
        return img

# --- 2. ENHANCED FEATURE EXTRACTOR ---
class IndustryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # WideResNet-101 provides deeper feature refinement for subtle defects
        backbone = models.wide_resnet101_2(weights='IMAGENET1K_V2')
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])
        self.layer2 = list(backbone.children())[5] # 512 channels
        self.layer3 = list(backbone.children())[6] # 1024 channels
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            # Spatial Alignment via bilinear interpolation
            x3_up = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=True)
            return torch.cat([x2, x3_up], dim=1) # 1536 fused channels

# --- 3. THE SIMPLENET MODULE ---
class SimpleNetPro(nn.Module):
    def __init__(self, in_channels=1536):
        super().__init__()
        # Adapter removes domain bias from pre-trained ImageNet features
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 512, 1) 
        )
        # Discriminator scores features based on learned decision boundary
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        adapted = self.adapter(x)
        b, c, h, w = adapted.shape
        flat = adapted.permute(0, 2, 3, 1).reshape(-1, c)
        return self.discriminator(flat).view(b, 1, h, w)

# --- 4. LOSS FUNCTION ---
def stable_anomaly_loss(pos_scores, neg_scores, margin=0.5):
    # Truncated L2 Loss prevents the model from "flipping" its logic
    loss_pos = torch.mean(torch.clamp(margin - pos_scores, min=0))
    loss_neg = torch.mean(torch.clamp(margin + neg_scores, min=0))
    return loss_pos + loss_neg

# --- 5. EXECUTION PIPELINE ---
def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    extractor = IndustryBackbone().to(device)
    model = SimpleNetPro().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Industry standard 512x512 resolution for minute crack detection
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BottleDataset("dataset/train/good", transform=data_transform), 
                              batch_size=4, shuffle=True)

    print("Training Optimized SimpleNet...")
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            features = extractor(batch)
            pos_scores = model(features)
            
            # Synthetic anomaly scale increased to 0.05 for clear boundary learning
            noise = torch.randn_like(features) * 0.01
            neg_scores = model(features + noise)
            
            loss = stable_anomaly_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0: print(f"Epoch {epoch} complete")

    print("\nEvaluating and Visualizing Results...")
    y_true, y_scores = [], []
    model.eval()
    
    test_paths = [(1, "dataset/test/structural_damage"), (0, "dataset/test/good")]
    for label, folder in test_paths:
        if not os.path.exists(folder): continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn))
            img_t = data_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Smooth score maps to reduce pixel-level noise
                score_map = torch.sigmoid(model(extractor(img_t))).cpu().numpy()[0,0]
                score_map = cv2.resize(score_map, (512, 512))
                score_map = gaussian_filter(score_map, sigma=4)
                
                y_scores.append(np.max(score_map))
                y_true.append(label)

    # Final ROC metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'SimpleNet (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.savefig("final_simplenet_roc.png")
    
    print(f"Final AUROC: {auroc:.4f}. Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()    