import numpy as np
import cv2
import os

# ==========================================
# Step 1: 10-bit YUV420 Data Loader
# ==========================================
def read_yuv420_10bit_frame(file_obj, width, height):
    """從二進位檔案中讀取單張 10-bit YUV420 影格"""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # 10-bit 數值在電腦中以 16-bit (uint16) 儲存
    y = np.fromfile(file_obj, dtype=np.uint16, count=y_size)
    if y.size == 0: 
        return None, None, None # 檔案讀取完畢 (EOF)
    y = y.reshape((height, width))

    u = np.fromfile(file_obj, dtype=np.uint16, count=uv_size)
    u = u.reshape((height // 2, width // 2))

    v = np.fromfile(file_obj, dtype=np.uint16, count=uv_size)
    v = v.reshape((height // 2, width // 2))

    return y, u, v

def write_yuv420_10bit_frame(file_obj, y, u, v):
    """將 YUV 陣列寫入二進位檔案"""
    y.astype(np.uint16).tofile(file_obj)
    u.astype(np.uint16).tofile(file_obj)
    v.astype(np.uint16).tofile(file_obj)

# ==========================================
# Step 2: Baseline 處理函數 (Bicubic Upsampling)
# ==========================================
def process_video(input_path, output_path, in_w=1920, in_h=1080, out_w=3840, out_h=2160):
    print(f"開始處理: {os.path.basename(input_path)}")
    
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        frame_idx = 0
        while True:
            # 讀取 FHD 奇數格
            y, u, v = read_yuv420_10bit_frame(fin, in_w, in_h)
            if y is None:
                break
            
            # 使用 Bicubic 放大至 4K (Baseline 核心)
            y_up = cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            u_up = cv2.resize(u, (out_w // 2, out_h // 2), interpolation=cv2.INTER_CUBIC)
            v_up = cv2.resize(v, (out_w // 2, out_h // 2), interpolation=cv2.INTER_CUBIC)
            
            # 寫出 4K 奇數格
            write_yuv420_10bit_frame(fout, y_up, u_up, v_up)
            frame_idx += 1
            print(f"  - 已完成第 {frame_idx} 幀...", end='\r')
            
    print(f"\n輸出完成: {os.path.basename(output_path)}\n")

# ==========================================
# 執行主程式
# ==========================================
if __name__ == '__main__':
    # 這裡以你剛剛貼的 AMS05 序列為例
    video_name = "H2_H3_AMS05"
    qps = ["27", "32", "37", "42"] # 4種壓縮率
    
    # 定義輸入與輸出的資料夾層級
    base_dir = "./bitstream/base"
    output_dir = "./my_output"
    
    for qp in qps:
        # 輸入檔名: odd_H2_H3_AMS05_27_0_5.layer0.yuv
        input_filename = f"odd_{video_name}_{qp}_0_5.layer0.yuv"
        # 輸出檔名: my_baseline_H2_H3_AMS05_27.yuv (取好認的名字)
        output_filename = f"my_baseline_{video_name}_{qp}.yuv"
        
        in_path = os.path.join(base_dir, input_filename)
        out_path = os.path.join(output_dir, output_filename)
        
        process_video(in_path, out_path)