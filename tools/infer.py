import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
import time
from tqdm import tqdm
from model.faster_rcnn import FasterRCNN
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader

# 1. ARM/MPS Support Fix
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    aps = []
    for idx, label in enumerate(gt_labels):
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True
        
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                prec_interp_pt = precisions[recalls >= interp_pt]
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
            
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset('test', im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    print(f"Loading checkpoint from {ckpt_path} to {device}...")
    faster_rcnn_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    return faster_rcnn_model, voc, test_dataset


# 2. Model Metrics Function (New Requirement)
def print_model_metrics(model, device):
    print("\n" + "="*30)
    print("MODEL PERFORMANCE METRICS")
    print("="*30)
    
    # 1. Model Size (MB) & Parameters
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model Size (Weights): {size_all_mb:.2f} MB")
    print(f"Total Parameters: {total_params/1e6:.2f} Million")

    # 2. Inference Speed (FPS)
    print("Calculating Inference FPS (Warmup + 50 runs)...")
    dummy_input = torch.randn(1, 3, 600, 600).to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10): 
            _ = model(dummy_input, None)
    
    # Timing
    t0 = time.time()
    num_runs = 50
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input, None)
    t1 = time.time()
    
    fps = num_runs / (t1 - t0)
    print(f"Inference Speed: {fps:.2f} FPS")
    print("="*30 + "\n")


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    
    faster_rcnn_model.roi_head.low_score_threshold = 0.7
    
    for sample_count in tqdm(range(10), desc="Generating Samples"):
        random_idx = random.randint(0, len(voc) - 1)
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()
        
        # Save GT
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=text, org=(x1+5, y1+15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('samples/output_frcnn_gt_{}.png'.format(sample_count), gt_im)
        
        # Prediction
        with torch.no_grad():
            rpn_output, frcnn_output = faster_rcnn_model(im, None)
            
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im_pred = cv2.imread(fname)
        im_copy = im_pred.copy()
        
        # Save Preds
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(im_pred, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            
            text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()], scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(im_pred, text=text, org=(x1+5, y1+15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1, fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            
        cv2.addWeighted(im_copy, 0.7, im_pred, 0.3, 0, im_pred)
        cv2.imwrite('samples/output_frcnn_{}.jpg'.format(sample_count), im_pred)


def evaluate_map(args):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    
    # Calculate and Print Model Metrics (Size, FPS)
    print_model_metrics(faster_rcnn_model, device)
    
    gts = []
    preds = []
    print("Starting mAP Evaluation...")
    
    with torch.no_grad():
        for im, target, fname in tqdm(test_dataset, desc="Evaluating"):
            im = im.float().to(device)
            target_boxes = target['bboxes'].float().to(device)[0]
            target_labels = target['labels'].long().to(device)[0]
            
            _, frcnn_output = faster_rcnn_model(im, None)

            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']
            
            pred_boxes = {}
            gt_boxes = {}
            for label_name in voc.label2idx:
                pred_boxes[label_name] = []
                gt_boxes[label_name] = []
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = labels[idx].detach().cpu().item()
                score = scores[idx].detach().cpu().item()
                label_name = voc.idx2label[label]
                pred_boxes[label_name].append([x1, y1, x2, y2, score])
                
            for idx, box in enumerate(target_boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = target_labels[idx].detach().cpu().item()
                label_name = voc.idx2label[label]
                gt_boxes[label_name].append([x1, y1, x2, y2])
            
            gts.append(gt_boxes)
            preds.append(pred_boxes)
   
    mean_ap, all_aps = compute_map(preds, gts, method='interp')
    print('\nClass Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        cls_name = voc.idx2label[idx]
        if cls_name in all_aps:
             print('AP for class {} = {:.4f}'.format(cls_name, all_aps[cls_name]))
    print('\nMean Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    # Changed type=bool to type=lambda x: (str(x).lower() == 'true') because type=bool doesn't work as expected in argparse
    parser.add_argument('--evaluate', dest='evaluate',
                        default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=False, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.infer_samples:
        infer(args)
    else:
        print('Skipping sample inference (use --infer_samples)')
        
    if args.evaluate:
        evaluate_map(args)
    else:
        print('Skipping evaluation (use --evaluate)')