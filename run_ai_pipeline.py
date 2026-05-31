# import numpy as np
# import cv2
# import os
# import torch
# import torch.nn as nn

# # ==========================================
# # 1. PyTorch 模型定義 (輕量殘差網路)
# # ==========================================
# class ResidualWarpNet(nn.Module):
#     def __init__(self, channels=16):
#         super(ResidualWarpNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
#         self.relu1 = nn.PReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu2 = nn.PReLU()
#         self.conv3 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
#         # [魔法機制] 零初始化
#         nn.init.zeros_(self.conv3.weight)
#         nn.init.zeros_(self.conv3.bias)
        
#     def forward(self, y_t_minus_1, y_t_vvc, y_t_plus_1):
#         x = torch.cat([y_t_minus_1, y_t_vvc, y_t_plus_1], dim=1)
#         x = self.relu1(self.conv1(x))
#         x = self.relu2(self.conv2(x))
#         residual = self.conv3(x)
#         return y_t_vvc + residual

# # ==========================================
# # 2. 核心演算法：光流對齊 (Optical Flow)
# # ==========================================
# def align_frames(img_ref, img_target, scale=0.25):
#     h, w = img_ref.shape
#     small_w, small_h = int(w * scale), int(h * scale)
    
#     ref_small = cv2.resize(img_ref, (small_w, small_h), interpolation=cv2.INTER_AREA)
#     tgt_small = cv2.resize(img_target, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
#     ref_8 = (np.clip(ref_small, 0, 1023) / 4).astype(np.uint8)
#     tgt_8 = (np.clip(tgt_small, 0, 1023) / 4).astype(np.uint8)
    
#     flow_small = cv2.calcOpticalFlowFarneback(tgt_8, ref_8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     flow_large = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    
#     grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#     map_x = (grid_x + flow_large[..., 0]).astype(np.float32)
#     map_y = (grid_y + flow_large[..., 1]).astype(np.float32)
    
#     warped = cv2.remap(img_ref.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
#     return warped

# # ==========================================
# # 3. YUV 讀取工具 (支援隨機抽幀)
# # ==========================================
# def read_yuv_frame_by_index(filepath, index, width=3840, height=2160):
#     frame_bytes = int(width * height * 1.5 * 2)
#     with open(filepath, 'rb') as f:
#         f.seek(index * frame_bytes)
#         y_size = width * height
#         uv_size = (width // 2) * (height // 2)
        
#         y = np.fromfile(f, dtype=np.uint16, count=y_size)
#         if y.size < y_size: return None, None, None
#         y = y.reshape((height, width))
        
#         u = np.fromfile(f, dtype=np.uint16, count=uv_size).reshape((height // 2, width // 2))
#         v = np.fromfile(f, dtype=np.uint16, count=uv_size).reshape((height // 2, width // 2))
#         return y, u, v

# # ==========================================
# # 4. 主推論管線 (Inference Pipeline)
# # ==========================================
# def process_video_ai(video_name, file_prefix, suffix, qp, frames_count=200):
#     print(f"\n🚀 開始處理 {video_name} (QP: {qp})")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 💡 使用動態對應的檔名邏輯
#     upscaled_path = f"./bitstream/upscaled/odd_{file_prefix}_{qp}_{suffix}_up.layer0.yuv"
#     enhance_path = f"./bitstream/enhance/even_{file_prefix}_{qp}_{suffix}.layer1.yuv"
#     out_path = f"./my_output/ai_{video_name}_{qp}.yuv"
    
#     model = ResidualWarpNet(channels=32).to(device)
#     ckpt_path = "/home/ddmanddman/AIDJ/CV_FP_0526/Release_v2/weights/resnet_ep28_loss0.0219.pth"
#     model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
#     model.eval()
    
#     with open(out_path, 'wb') as fout:
#         for i in range(frames_count):
#             print(f"  - 正在處理第 {i} 幀 (光流對齊 + CNN運算)...", end='\r')
            
#             y_t, u_t, v_t = read_yuv_frame_by_index(upscaled_path, i)
#             y_t_minus_1, _, _ = read_yuv_frame_by_index(enhance_path, i)
#             next_idx = i + 1 if i + 1 < frames_count else i 
#             y_t_plus_1, _, _ = read_yuv_frame_by_index(enhance_path, next_idx)
            
#             if y_t is None or y_t_minus_1 is None or y_t_plus_1 is None:
#                 break
                
#             warped_minus_1 = align_frames(y_t_minus_1, y_t)
#             warped_plus_1 = align_frames(y_t_plus_1, y_t)
            
#             t_minus_1 = torch.from_numpy(warped_minus_1).float().unsqueeze(0).unsqueeze(0) / 1023.0
#             t_vvc = torch.from_numpy(y_t).float().unsqueeze(0).unsqueeze(0) / 1023.0
#             t_plus_1 = torch.from_numpy(warped_plus_1).float().unsqueeze(0).unsqueeze(0) / 1023.0
            
#             t_minus_1 = t_minus_1.to(device)
#             t_vvc = t_vvc.to(device)
#             t_plus_1 = t_plus_1.to(device)
            
#             with torch.no_grad():
#                 out_y_tensor = model(t_minus_1, t_vvc, t_plus_1)
                
#             out_y = (out_y_tensor.cpu().squeeze().numpy() * 1023.0).clip(0, 1023).astype(np.uint16)
            
#             out_y.tofile(fout)
#             u_t.astype(np.uint16).tofile(fout)
#             v_t.astype(np.uint16).tofile(fout)
            
#     print(f"\n✅ {video_name} (QP: {qp}) 完成！")

# if __name__ == '__main__':
#     # 💡 建立完美的檔案名稱對應表 (Config)
#     video_configs = [
#         {"video_name": "Procession", "file_prefix": "Procession", "suffix": "0_4", "qps": ["25", "30", "35", "40"]},
#         {"video_name": "H2_H3_AMS05", "file_prefix": "H2_H3_AMS05", "suffix": "0_5", "qps": ["27", "32", "37", "42"]},
#         {"video_name": "Zombie", "file_prefix": "ZombieClimbing2", "suffix": "0_4", "qps": ["27", "32", "37", "42"]},
#         {"video_name": "WalkInPark", "file_prefix": "H2_WalkInPark", "suffix": "0_4", "qps": ["27", "32", "37", "42"]}
#     ]
    
#     for config in video_configs:
#         for qp in config["qps"]:
#             process_video_ai(
#                 video_name=config["video_name"],
#                 file_prefix=config["file_prefix"],
#                 suffix=config["suffix"],
#                 qp=qp,
#                 frames_count=200
#             )


#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

@dataclass(frozen=True)
class SequenceInfo:
    name: str
    bitstream_key: str
    org_file: str
    qps: Tuple[str, ...]
    suffix: str
    fps: int

SEQUENCES: Dict[str, SequenceInfo] = {
    "Zombie": SequenceInfo(name="Zombie", bitstream_key="ZombieClimbing2", org_file="odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv", qps=("27", "32", "37", "42"), suffix="0_4", fps=24),
    "WalkInPark": SequenceInfo(name="WalkInPark", bitstream_key="H2_WalkInPark", org_file="odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv", qps=("27", "32", "37", "42"), suffix="0_4", fps=60),
    "Procession": SequenceInfo(name="Procession", bitstream_key="Procession", org_file="odd_Procession_3840x2160_60fps_10bit_420.yuv", qps=("25", "30", "35", "40"), suffix="0_4", fps=60),
    "AMS05": SequenceInfo(name="AMS05", bitstream_key="H2_H3_AMS05", org_file="odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv", qps=("27", "32", "37", "42"), suffix="0_5", fps=60),
}

# ==========================================
# 🚀 這裡換成妳的 20 小時極限大腦！
# ==========================================
class ResidualWarpNet_Max(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
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

    def forward(self, y_t_minus_1, y_t_anchor, y_t_plus_1):
        x = torch.cat([y_t_minus_1, y_t_anchor, y_t_plus_1], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        residual = self.conv5(x)
        return y_t_anchor + residual

# -----------------------------------------------------------------------------
# 隊友的優秀工具函數 (保持不變)
# -----------------------------------------------------------------------------
def frame_bytes(width: int, height: int) -> int:
    return (width * height + 2 * (width // 2) * (height // 2)) * 2

def count_yuv_frames(path: str | Path, width: int, height: int) -> int:
    path = Path(path)
    b = frame_bytes(width, height)
    size = path.stat().st_size
    return size // b

def read_yuv_frame_by_index(filepath: str | Path, index: int, width: int = 3840, height: int = 2160):
    filepath = Path(filepath)
    b = frame_bytes(width, height)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    with filepath.open("rb") as f:
        f.seek(index * b)
        y = np.fromfile(f, dtype="<u2", count=y_size)
        if y.size < y_size: return None, None, None
        u = np.fromfile(f, dtype="<u2", count=uv_size)
        v = np.fromfile(f, dtype="<u2", count=uv_size)
        if u.size < uv_size or v.size < uv_size: return None, None, None
    return (y.reshape((height, width)) & 0x03FF,
            u.reshape((height // 2, width // 2)) & 0x03FF,
            v.reshape((height // 2, width // 2)) & 0x03FF)

def write_yuv420p10_frame(f, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    np.clip(y, 0, 1023).astype("<u2", copy=False).tofile(f)
    np.clip(u, 0, 1023).astype("<u2", copy=False).tofile(f)
    np.clip(v, 0, 1023).astype("<u2", copy=False).tofile(f)

def upsample_base_frame_to_4k(y, u, v):
    y4 = cv2.resize(y.astype(np.float32), (3840, 2160), interpolation=cv2.INTER_CUBIC)
    u4 = cv2.resize(u.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_CUBIC)
    v4 = cv2.resize(v.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return (np.clip(y4, 0, 1023).astype(np.uint16), np.clip(u4, 0, 1023).astype(np.uint16), np.clip(v4, 0, 1023).astype(np.uint16))

def _to_8bit_for_flow(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img.astype(np.float32) / 4.0), 0, 255).astype(np.uint8)

def align_frames(img_ref: np.ndarray, img_target: np.ndarray, scale: float = 0.25) -> np.ndarray:
    h, w = img_ref.shape
    small_w, small_h = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
    ref_small = cv2.resize(img_ref, (small_w, small_h), interpolation=cv2.INTER_AREA)
    tgt_small = cv2.resize(img_target, (small_w, small_h), interpolation=cv2.INTER_AREA)
    ref_8 = _to_8bit_for_flow(ref_small)
    tgt_8 = _to_8bit_for_flow(tgt_small)
    flow_small = cv2.calcOpticalFlowFarneback(tgt_8, ref_8, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow_large = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) / float(scale)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_large[..., 0].astype(np.float32)
    map_y = grid_y + flow_large[..., 1].astype(np.float32)
    warped = cv2.remap(img_ref.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return warped

def get_paths(release_root, seq, qp, anchor_source="upscaled"):
    root = Path(release_root)
    info = SEQUENCES[seq]
    key, suffix = info.bitstream_key, info.suffix
    paths = {
        "base": root / "bitstream" / "base" / f"odd_{key}_{qp}_{suffix}.layer0.yuv",
        "upscaled": root / "bitstream" / "upscaled" / f"odd_{key}_{qp}_{suffix}_up.layer0.yuv",
        "enhance": root / "bitstream" / "enhance" / f"even_{key}_{qp}_{suffix}.layer1.yuv",
        "org": root / "orgYUV" / info.org_file,
    }
    paths["anchor"] = paths[anchor_source]
    return paths

def load_checkpoint(model, ckpt_path, device, strict=True):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    clean = OrderedDict()
    for k, v in ckpt.items():
        clean[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    model.load_state_dict(clean, strict=strict)

def _weight_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 2 else np.ones(h)
    wx = np.hanning(w) if w > 2 else np.ones(w)
    return np.outer(np.maximum(wy, 1e-3), np.maximum(wx, 1e-3)).astype(np.float32)

def infer_y_tiled(model, device, y_m1, y_anchor, y_p1, tile=640, overlap=64, residual_scale=1.0):
    h, w = y_anchor.shape
    out = np.zeros((h, w), dtype=np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile: ys.append(max(0, h - tile))
    if xs[-1] != w - tile: xs.append(max(0, w - tile))

    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                hm, wm = y1 - y0, x1 - x0
                arrs = [y_m1[y0:y1, x0:x1], y_anchor[y0:y1, x0:x1], y_p1[y0:y1, x0:x1]]
                tens = [torch.from_numpy(a.astype(np.float32) / 1023.0).unsqueeze(0).unsqueeze(0).to(device) for a in arrs]
                pred = model(tens[0], tens[1], tens[2])
                if residual_scale != 1.0:
                    pred = tens[1] + residual_scale * (pred - tens[1])
                pred_np = (pred.squeeze().detach().cpu().numpy() * 1023.0).astype(np.float32)
                win = _weight_window(hm, wm)
                out[y0:y1, x0:x1] += pred_np * win
                acc[y0:y1, x0:x1] += win
    out = out / np.maximum(acc, 1e-6)
    return np.clip(out, 0, 1023).astype(np.uint16)

def process_one_file(release_root, seq, qp, checkpoint, output_dir, anchor_source="upscaled", flow_scale=0.25, tile=640, overlap=64, residual_scale=1.0, device_name="auto"):
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    paths = get_paths(release_root, seq, qp, anchor_source=anchor_source)
    
    # 🚀 這裡掛載極限版大腦
    model = ResidualWarpNet_Max(channels=64).to(device)
    load_checkpoint(model, checkpoint, device, strict=True)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_{seq}_{qp}.yuv"

    enh_count = count_yuv_frames(paths["enhance"], 3840, 2160)
    anchor_count = count_yuv_frames(paths["anchor"], 3840, 2160) if anchor_source == "upscaled" else count_yuv_frames(paths["anchor"], 1920, 1080)
    out_count = min(anchor_count, enh_count - 1)
    print(f"[INFO] {seq} QP{qp}: frames={out_count}, device={device}")

    with out_path.open("wb") as fout:
        for i in range(out_count):
            if anchor_source == "upscaled":
                y_anchor, u_anchor, v_anchor = read_yuv_frame_by_index(paths["anchor"], i, 3840, 2160)
            else:
                yb, ub, vb = read_yuv_frame_by_index(paths["anchor"], i, 1920, 1080)
                y_anchor, u_anchor, v_anchor = upsample_base_frame_to_4k(yb, ub, vb)
            y_m1, _, _ = read_yuv_frame_by_index(paths["enhance"], i, 3840, 2160)
            y_p1, _, _ = read_yuv_frame_by_index(paths["enhance"], i + 1, 3840, 2160)
            
            warped_m1 = align_frames(y_m1, y_anchor, scale=flow_scale)
            warped_p1 = align_frames(y_p1, y_anchor, scale=flow_scale)
            out_y = infer_y_tiled(model, device, warped_m1, y_anchor, warped_p1, tile=tile, overlap=overlap, residual_scale=residual_scale)
            write_yuv420p10_frame(fout, out_y, u_anchor, v_anchor)
            if i % 10 == 0 or i == out_count - 1:
                print(f"[INFO] {seq} QP{qp}: frame {i+1}/{out_count} done", flush=True)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default=".")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", default="my_output")
    ap.add_argument("--sequences", nargs="+", default=["all"])
    args = ap.parse_args()
    
    seqs = ["Zombie", "WalkInPark", "Procession", "AMS05"] if "all" in args.sequences else args.sequences
    for seq in seqs:
        for qp in SEQUENCES[seq].qps:
            process_one_file(args.release_root, seq, qp, args.checkpoint, args.output_dir)

if __name__ == "__main__":
    main()