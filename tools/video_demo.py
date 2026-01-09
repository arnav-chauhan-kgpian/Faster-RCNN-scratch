import torch
import cv2
import argparse
import yaml
import time
import os
import numpy as np
from model.faster_rcnn import FasterRCNN

# 1. Device Setup (ARM/x86 Compatible)
def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()

# Standard PASCAL VOC Classes (Hardcoded to avoid loading the full dataset)
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
    'sofa', 'train', 'tvmonitor'
]

def run_video(args):
    # 2. Load Config & Model
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)
    
    # Initialize Model
    model = FasterRCNN(config['model_params'], num_classes=config['dataset_params']['num_classes'])
    
    # Load Weights
    ckpt_path = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Train the model first.")
        return

    print(f"Loading weights from {ckpt_path} to {device}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Fallback if FPS is 0 (common with webcams)
    if fps == 0: fps = 20

    # Setup Video Writer
    save_path = "output_detection.mp4"
    # 'avc1' is often more compatible with Mac/QuickTime than 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'avc1') if os.name == 'posix' else cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print(f"Processing video resolution {width}x{height} at {fps} FPS...")
    
    frame_count = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # ---------------------------------------------------------
            # CRITICAL FIX 1: Convert BGR (OpenCV) to RGB (Model Expectation)
            # ---------------------------------------------------------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------------
            # CRITICAL FIX 2: Normalize [0, 255] -> [0.0, 1.0]
            # ---------------------------------------------------------
            img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
            img_tensor /= 255.0  # Normalize to 0-1
            img_tensor = img_tensor.to(device).unsqueeze(0) # Batch size 1
            
            # Inference
            t0 = time.time()
            _, frcnn_output = model(img_tensor, None)
            t1 = time.time()
            
            # Draw Detections
            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']
            
            for box, label, score in zip(boxes, labels, scores):
                if score < args.threshold: continue
                
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
                # Get class name safely
                lbl_idx = label.item()
                cls_name = VOC_CLASSES[lbl_idx] if lbl_idx < len(VOC_CLASSES) else str(lbl_idx)
                
                # Draw Box (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label (White Text on Green Background)
                label_text = f"{cls_name}: {score:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # FPS Counter
            inference_fps = 1 / (t1 - t0)
            cv2.putText(frame, f"FPS: {inference_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            writer.write(frame)
            
            # Optional: Display (Comment out if running on headless server)
            # cv2.imshow('Detection', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done! Video saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/voc.yaml')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video (mp4/avi) or 0 for webcam')
    parser.add_argument('--threshold', type=float, default=0.7)
    args = parser.parse_args()
    run_video(args)