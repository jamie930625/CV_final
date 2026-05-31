import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import time

# 引入模型與工具 (確保 run_ai_pipeline.py 在同一資料夾)
from run_ai_pipeline import ResidualWarpNet, align_frames, read_yuv_frame_by_index

# ==========================================
# 1. PyTorch Dataset (支援 Multi-QP 與優化版光流)
# ==========================================
class FramePatchDataset(Dataset):
    def __init__(self, qps=["27", "32", "37", "42"], video_name="H2_H3_AMS05", frames_count=49, patch_size=128, samples_per_frame=15):
        self.frames_count = frames_count
        self.patch_size = patch_size
        self.samples_per_frame = samples_per_frame
        
        # 將所有 QP 的路徑整理好
        self.paths = []
        for qp in qps:
            self.paths.append({
                'up': f"./bitstream/upscaled/odd_{video_name}_{qp}_0_5_up.layer0.yuv",
                'en': f"./bitstream/enhance/even_{video_name}_{qp}_0_5.layer1.yuv",
                'org': f"./orgYUV/odd_{video_name}_3840x2160_10bit_420_HLG.yuv"
            })
            
    def __len__(self):
        # 總資料量 = 幀數 * 每幀抽樣數 * 壓縮率種類
        return self.frames_count * self.samples_per_frame * len(self.paths)

    def __getitem__(self, idx):
        # 隨機挑選一個壓縮率的資料來訓練
        path_group = random.choice(self.paths)
        frame_idx = random.randint(0, self.frames_count - 1)
        
        # 讀取影像 (只取 Y 通道)
        y_t, _, _ = read_yuv_frame_by_index(path_group['up'], frame_idx)
        y_t_minus_1, _, _ = read_yuv_frame_by_index(path_group['en'], frame_idx)
        next_idx = frame_idx + 1 if frame_idx + 1 < self.frames_count else frame_idx
        y_t_plus_1, _, _ = read_yuv_frame_by_index(path_group['en'], next_idx)
        y_target, _, _ = read_yuv_frame_by_index(path_group['org'], frame_idx)
        
        # 若發生意外讀取失敗，直接回傳全零陣列防呆
        if y_t is None or y_target is None:
            shape = (1, self.patch_size, self.patch_size)
            return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
        
        # 💡 使用縮小版光流對齊 (解決算太久的問題)
        warped_minus_1 = align_frames(y_t_minus_1, y_t, scale=0.25)
        warped_plus_1 = align_frames(y_t_plus_1, y_t, scale=0.25)
        
        # 隨機裁切 (Random Crop)
        h, w = y_t.shape
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        def crop(img): return img[top:top+self.patch_size, left:left+self.patch_size]
            
        patch_minus_1 = crop(warped_minus_1)
        patch_t = crop(y_t)
        patch_plus_1 = crop(warped_plus_1)
        patch_target = crop(y_target)
        
        # 轉成 PyTorch Tensor 並 Normalize (0~1)
        t_m1 = torch.from_numpy(patch_minus_1).float().unsqueeze(0) / 1023.0
        t_vvc = torch.from_numpy(patch_t).float().unsqueeze(0) / 1023.0
        t_p1 = torch.from_numpy(patch_plus_1).float().unsqueeze(0) / 1023.0
        t_tgt = torch.from_numpy(patch_target).float().unsqueeze(0) / 1023.0
        
        return t_m1, t_vvc, t_p1, t_tgt

# ==========================================
# 2. 徹夜掛機主迴圈 (Over-Night Training Loop)
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 啟動訓練裝置: {device}")
    
    # 建立模型與 DataLoader
    # 將 Channel 從 16 提升到 32，給模型多一點學習能力 (MACs 仍在安全範圍內)
    model = ResidualWarpNet(channels=32).to(device)
    dataset = FramePatchDataset(patch_size=128, samples_per_frame=20)
    
    # num_workers 設定為 4 可以加速資料讀取，但如果實驗室機器會報錯，可改回 0
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True) 
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # 每過 10 個 Epoch，學習率衰減一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 建立資料夾存放權重
    os.makedirs("./weights", exist_ok=True)
    
    # 睡覺設定：跑 30 個 Epoch，大約可以跑上幾個小時
    epochs = 30
    print(f"🚀 開始掛機訓練... 預計執行 {epochs} Epochs")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (t_m1, t_vvc, t_p1, target) in enumerate(dataloader):
            t_m1, t_vvc = t_m1.to(device), t_vvc.to(device)
            t_p1, target = t_p1.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(t_m1, t_vvc, t_p1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 每 50 個 Batch 印一次進度
            if batch_idx % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")
                
        scheduler.step() # 更新學習率
        
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"==== Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.6f} | 耗時: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f} ====\n")
        
        # 存檔：檔名包含 Epoch 數與 Loss，方便明天挑選
        save_path = f"./weights/resnet_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"
        torch.save(model.state_dict(), save_path)
        
    print("🎉 訓練大功告成！早安！")

if __name__ == '__main__':
    train_model()