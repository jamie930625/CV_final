import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import time

# 💡 防死鎖設定
cv2.setNumThreads(0) 

from run_ai_pipeline import align_frames, read_yuv_frame_by_index

# ==========================================
# ==========================================
class ResidualWarpNet_Max(nn.Module):
    def __init__(self, channels=64):
        super(ResidualWarpNet_Max, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        nn.init.zeros_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)
        
    def forward(self, y_t_minus_1, y_t_vvc, y_t_plus_1):
        x = torch.cat([y_t_minus_1, y_t_vvc, y_t_plus_1], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        residual = self.conv5(x)
        return y_t_vvc + residual

# ==========================================
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

# ==========================================
# ==========================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        # 計算 X 方向與 Y 方向的梯度 (邊緣)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        loss_dx = torch.mean(torch.abs(pred_dx - tgt_dx))
        loss_dy = torch.mean(torch.abs(pred_dy - tgt_dy))
        return loss_dx + loss_dy

# ==========================================
# ==========================================
class UniversalDataset(Dataset):
    def __init__(self, patch_size=128, samples_per_epoch=4000):
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.qps = ["27", "32", "37", "42"]
        
        self.configs = [
            {"prefix": "Procession", "suffix": "0_4", "qps": ["25", "30", "35", "40"], "org_key": "Procession"},
            {"prefix": "H2_H3_AMS05", "suffix": "0_5", "qps": ["27", "32", "37", "42"], "org_key": "AMS05"},
            {"prefix": "ZombieClimbing2", "suffix": "0_4", "qps": ["27", "32", "37", "42"], "org_key": "Zombie"},
            {"prefix": "H2_WalkInPark", "suffix": "0_4", "qps": ["27", "32", "37", "42"], "org_key": "WalkInPark"}
        ]
        
        self.org_files = {}
        for f in os.listdir("./orgYUV"):
            for config in self.configs:
                if config["org_key"] in f:
                    self.org_files[config["org_key"]] = f"./orgYUV/{f}"

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        config = random.choice(self.configs)
        qp = random.choice(config["qps"])
        frame_idx = random.randint(0, 48) 
        
        up_path = f"./bitstream/upscaled/odd_{config['prefix']}_{qp}_{config['suffix']}_up.layer0.yuv"
        en_path = f"./bitstream/enhance/even_{config['prefix']}_{qp}_{config['suffix']}.layer1.yuv"
        org_path = self.org_files.get(config["org_key"], None)
        
        y_t, _, _ = read_yuv_frame_by_index(up_path, frame_idx)
        y_t_m1, _, _ = read_yuv_frame_by_index(en_path, frame_idx)
        y_t_p1, _, _ = read_yuv_frame_by_index(en_path, frame_idx + 1)
        
        if y_t is None or org_path is None or y_t_m1 is None or y_t_p1 is None:
            shape = (1, self.patch_size, self.patch_size)
            return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
            
        y_target, _, _ = read_yuv_frame_by_index(org_path, frame_idx)
        
        # 裁切
        h, w = y_t.shape
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        def crop(img): return img[top:top+self.patch_size, left:left+self.patch_size]
            
        patch_t = crop(y_t)
        patch_m1 = crop(y_t_m1)
        patch_p1 = crop(y_t_p1)
        patch_tgt = crop(y_target)
        
        # 🚀 升級：資料擴增 (Data Augmentation) - 隨機水平與垂直翻轉
        if random.random() > 0.5:
            patch_t = np.flip(patch_t, axis=1).copy()
            patch_m1 = np.flip(patch_m1, axis=1).copy()
            patch_p1 = np.flip(patch_p1, axis=1).copy()
            patch_tgt = np.flip(patch_tgt, axis=1).copy()
        if random.random() > 0.5:
            patch_t = np.flip(patch_t, axis=0).copy()
            patch_m1 = np.flip(patch_m1, axis=0).copy()
            patch_p1 = np.flip(patch_p1, axis=0).copy()
            patch_tgt = np.flip(patch_tgt, axis=0).copy()
        
        warped_m1 = align_frames(patch_m1, patch_t, scale=1.0)
        warped_p1 = align_frames(patch_p1, patch_t, scale=1.0)
        
        t_m1 = torch.from_numpy(warped_m1).float().unsqueeze(0) / 1023.0
        t_vvc = torch.from_numpy(patch_t).float().unsqueeze(0) / 1023.0
        t_p1 = torch.from_numpy(warped_p1).float().unsqueeze(0) / 1023.0
        t_tgt = torch.from_numpy(patch_tgt).float().unsqueeze(0) / 1023.0
        
        return t_m1, t_vvc, t_p1, t_tgt

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 啟動 0531 過夜特訓版 (SR Charbonnier + Edge Loss)")
    
    model = ResidualWarpNet_Max(channels=64).to(device)
    dataset = UniversalDataset(samples_per_epoch=4000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0) 
    
    # 🚀 載入特製的 SR 專屬 Loss
    criterion_char = CharbonnierLoss().to(device)
    criterion_grad = GradientLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    epochs = 150 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    output_dir = "/home/ddmanddman/AIDJ/CV_FP_0526/Release_v2/0531過夜output"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch_idx, (t_m1, t_vvc, t_p1, target) in enumerate(dataloader):
            t_m1, t_vvc, t_p1, target = t_m1.to(device), t_vvc.to(device), t_p1.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 預測
            pred = model(t_m1, t_vvc, t_p1)
            
            # 💡 聯合損失計算：整體畫質(Charbonnier) + 0.05權重的邊緣銳化(Gradient)
            loss_char = criterion_char(pred, target)
            loss_edge = criterion_grad(pred, target)
            loss = loss_char + 0.05 * loss_edge
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Joint Loss: {loss.item():.6f}")
                
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        curr_lr = scheduler.get_last_lr()[0]
        print(f"==== 👑 0531 SR Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {curr_lr:.6f} | Time: {time.time()-start_time:.1f}s ====")
        
        save_path = os.path.join(output_dir, f"SR_Loss_ep{epoch+1:03d}_loss{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    train_model()
