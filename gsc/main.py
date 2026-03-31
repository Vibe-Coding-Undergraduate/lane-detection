import cv2
import numpy as np
import os

# ==========================================
# 1. 全局参数配置 (在这里一键调优)
# ==========================================
CONFIG = {
    # 数据路径
    "input_video": "data/test_video.mp4",
    "output_path": "output/result_v2.mp4",
    
    # Canny 边缘检测参数
    "canny_low": 50,
    "canny_high": 120,
    "gaussian_ksize": (5, 5),
    
    # ROI 区域系数 (基于图像宽高的比例)
    # 顺序：左下, 左上, 右上, 右侧切入点, 右下
    "roi_points": [
        (0.00, 1.00), # 左下
        (0.40, 0.50), # 左上
        (0.60, 0.50), # 右上
        (1.00, 0.65), # 右侧壁 (接住侧面进来的线)
        (1.00, 1.00)  # 右下
    ],
    
    # 霍夫变换参数
    "hough_rho": 1,
    "hough_theta": np.pi / 180,
    "hough_threshold": 20,
    "hough_min_len": 15,
    "hough_max_gap": 300,
    
    # 过滤逻辑
    "slope_min": 0.5,  # 约 26度
    "slope_max": 2.0,  # 约 63度
    
    # 调试模式 (设置为 True 会在视频中显示白色遮罩)
    "debug_mode": True 
}

# ==========================================
# 2. 核心处理函数
# ==========================================
def process_frame(frame):
    h, w = frame.shape[:2]
    
    # --- 步骤 A: 预处理 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, CONFIG["gaussian_ksize"], 0)
    edges = cv2.Canny(blur, CONFIG["canny_low"], CONFIG["canny_high"])
    
    # --- 步骤 B: ROI 掩膜 ---
    mask = np.zeros_like(edges)
    # 将 CONFIG 中的比例系数转换为实际像素坐标
    vertices = np.array([[
        (int(w * p[0]), int(h * p[1])) for p in CONFIG["roi_points"]
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # --- 步骤 C: 霍夫变换 ---
    line_img = np.zeros_like(frame)
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=CONFIG["hough_rho"], 
        theta=CONFIG["hough_theta"], 
        threshold=CONFIG["hough_threshold"], 
        minLineLength=CONFIG["hough_min_len"], 
        maxLineGap=CONFIG["hough_max_gap"]
    )
    
    # --- 步骤 D: 绘制与过滤 ---
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            slope = (y2 - y1) / (x2 - x1)
            
            # 使用配置中的斜率范围进行过滤
            if CONFIG["slope_min"] < abs(slope) < CONFIG["slope_max"]:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
                
    # --- 步骤 E: 叠加 ---
    result = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)
    
    # 如果开启调试模式，叠加半透明遮罩
    if CONFIG["debug_mode"]:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 1.0, mask_bgr, 0.2, 0)
        
    return result

# ==========================================
# 3. 视频流流水线
# ==========================================
def run_video_pipeline():
    if not os.path.exists('output'):
        os.makedirs('output')

    cap = cv2.VideoCapture(CONFIG["input_video"])
    if not cap.isOpened():
        print(f"无法打开视频: {CONFIG['input_video']}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(CONFIG["output_path"], fourcc, fps, (w, h))

    print(f"正在处理视频: {CONFIG['input_video']} -> {CONFIG['output_path']}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()
    print("处理圆满完成！")

if __name__ == "__main__":
    run_video_pipeline()
