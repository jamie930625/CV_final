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
# 隊友的工具函數
# -----------------------------------------------------------------------------
def frame_bytes(width: int, height: int) -> int:
    return (width * height + 2 * (width // 2) * (height // 2)) * 2

def count_yuv_frames(path: str | Path, width: int, height: int) -> int:
    path = Path(path)
    return path.stat().st_size // frame_bytes(width, height)

def read_yuv_frame_by_index(filepath: str | Path, index: int, width: int = 3840, height: int = 2160):
    b = frame_bytes(width, height)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    with Path(filepath).open("rb") as f:
        f.seek(index * b)
        y = np.fromfile(f, dtype="<u2", count=y_size)
        if y.size < y_size: return None, None, None
        u = np.fromfile(f, dtype="<u2", count=uv_size)
        v = np.fromfile(f, dtype="<u2", count=uv_size)
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


def align_frames_ultimate(img_ref: np.ndarray, img_target: np.ndarray, u_ref: np.ndarray=None, v_ref: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    回傳: (Warped_Y, Warped_U, Warped_V, Flow_large)
    """
    h, w = img_ref.shape
    
    # 套用雙邊濾波器抹平 128x128 區塊，保留真實邊緣，用來算光流最準
    ref_8 = _to_8bit_for_flow(img_ref)
    tgt_8 = _to_8bit_for_flow(img_target)
    ref_clean = cv2.bilateralFilter(ref_8, d=9, sigmaColor=75, sigmaSpace=75)
    tgt_clean = cv2.bilateralFilter(tgt_8, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 計算差異圖，判定大動態前景與微動態背景
    diff = cv2.absdiff(ref_clean, tgt_clean)
    _, mask_foreground = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mask_foreground = cv2.GaussianBlur(mask_foreground, (21, 21), 0).astype(np.float32) / 255.0
    mask_background = 1.0 - mask_foreground

    # 階段一：前景大動態 (Coarse-to-Fine)
    coarse_scale = 0.125
    cw, ch = int(w * coarse_scale), int(h * coarse_scale)
    ref_coarse = cv2.resize(ref_clean, (cw, ch), interpolation=cv2.INTER_AREA)
    tgt_coarse = cv2.resize(tgt_clean, (cw, ch), interpolation=cv2.INTER_AREA)
    
    flow_coarse = cv2.calcOpticalFlowFarneback(
        tgt_coarse, ref_coarse, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    fine_scale = 0.5 
    fw, fh = int(w * fine_scale), int(h * fine_scale)
    flow_coarse_up = cv2.resize(flow_coarse, (fw, fh), interpolation=cv2.INTER_LINEAR) * (fine_scale / coarse_scale)
    
    ref_fine = cv2.resize(ref_clean, (fw, fh), interpolation=cv2.INTER_AREA)
    tgt_fine = cv2.resize(tgt_clean, (fw, fh), interpolation=cv2.INTER_AREA)
    
    flow_fine = cv2.calcOpticalFlowFarneback(
        tgt_fine, ref_fine, flow_coarse_up, pyr_scale=0.5, levels=3, winsize=11, iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW
    )
    flow_large_foreground = cv2.resize(flow_fine, (w, h), interpolation=cv2.INTER_LINEAR) / fine_scale

    # 階段二：背景微動態 (High-Res Local)
    # 直接在原圖(或 1080p)算微小光流，避免糊掉 WalkInPark 的樹葉
    flow_local = cv2.calcOpticalFlowFarneback(
        tgt_clean, ref_clean, None, pyr_scale=0.5, levels=3, winsize=11, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    
    # 融合光流場：大動態吃前景光流，微動態吃背景光流
    flow_large = np.zeros_like(flow_large_foreground)
    flow_large[..., 0] = flow_large_foreground[..., 0] * mask_foreground + flow_local[..., 0] * mask_background
    flow_large[..., 1] = flow_large_foreground[..., 1] * mask_foreground + flow_local[..., 1] * mask_background

    # 套用光流 Warp Y 通道 (注意！這裡是 Warp 原始的 10bit 未濾波圖)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_large[..., 0]
    map_y = grid_y + flow_large[..., 1]
    warped_y = cv2.remap(img_ref.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    if u_ref is not None and v_ref is not None:
        # 4:2:0 格式，光流場縮小一半
        flow_uv = cv2.resize(flow_large, (w//2, h//2), interpolation=cv2.INTER_LINEAR) / 2.0
        grid_x_uv, grid_y_uv = np.meshgrid(np.arange(w//2, dtype=np.float32), np.arange(h//2, dtype=np.float32))
        map_x_uv = grid_x_uv + flow_uv[..., 0]
        map_y_uv = grid_y_uv + flow_uv[..., 1]
        
        warped_u = cv2.remap(u_ref.astype(np.float32), map_x_uv, map_y_uv, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        warped_v = cv2.remap(v_ref.astype(np.float32), map_x_uv, map_y_uv, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    else:
        warped_u, warped_v = None, None

    return warped_y, warped_u, warped_v, flow_large

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

def infer_y_tiled(model, device, y_m1, y_anchor, y_p1, tile=1024, overlap=64, residual_scale=1.0):
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
                pred_np = (pred.squeeze().detach().cpu().numpy() * 1023.0).astype(np.float32)
                win = _weight_window(hm, wm)
                out[y0:y1, x0:x1] += pred_np * win
                acc[y0:y1, x0:x1] += win
    out = out / np.maximum(acc, 1e-6)
    return np.clip(out, 0, 1023).astype(np.uint16)

def process_one_file(release_root, seq, qp, checkpoint, output_dir, anchor_source="upscaled", tile=1024, overlap=64, residual_scale=1.0, device_name="auto"):
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    paths = get_paths(release_root, seq, qp, anchor_source=anchor_source)
    
    model = ResidualWarpNet_Max(channels=64).to(device)
    load_checkpoint(model, checkpoint, device, strict=True)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_{seq}_{qp}.yuv"

    enh_count = count_yuv_frames(paths["enhance"], 3840, 2160)
    anchor_count = count_yuv_frames(paths["anchor"], 3840, 2160) if anchor_source == "upscaled" else count_yuv_frames(paths["anchor"], 1920, 1080)
    out_count = min(anchor_count, enh_count - 1)
    print(f"[INFO] {seq} QP{qp}: frames={out_count}, method=Ultimate (Artifact+Masking+UV)")

    with out_path.open("wb") as fout:
        for i in range(out_count):
            if anchor_source == "upscaled":
                y_anchor, u_anchor, v_anchor = read_yuv_frame_by_index(paths["anchor"], i, 3840, 2160)
            else:
                yb, ub, vb = read_yuv_frame_by_index(paths["anchor"], i, 1920, 1080)
                y_anchor, u_anchor, v_anchor = upsample_base_frame_to_4k(yb, ub, vb)
                
            y_m1, u_m1, v_m1 = read_yuv_frame_by_index(paths["enhance"], i, 3840, 2160)
            y_p1, u_p1, v_p1 = read_yuv_frame_by_index(paths["enhance"], i + 1, 3840, 2160)
            
            wy_m1, wu_m1, wv_m1, _ = align_frames_ultimate(y_m1, y_anchor, u_m1, v_m1)
            wy_p1, wu_p1, wv_p1, _ = align_frames_ultimate(y_p1, y_anchor, u_p1, v_p1)
            
            # CNN 推論 Y 通道
            out_y = infer_y_tiled(model, device, wy_m1, y_anchor, wy_p1, tile=tile, overlap=overlap)
            
            # 使用 70% 高畫質對齊色彩 + 30% 模糊底圖色彩，確保不會因為光流微小誤差而出現色彩斷層
            out_u = (wu_m1 * 0.35 + wu_p1 * 0.35 + u_anchor * 0.30).astype(np.uint16)
            out_v = (wv_m1 * 0.35 + wv_p1 * 0.35 + v_anchor * 0.30).astype(np.uint16)
            
            write_yuv420p10_frame(fout, out_y, out_u, out_v)
            
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
