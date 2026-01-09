import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Import your custom modules
from model.faster_rcnn import FasterRCNN
from dataset.voc import VOCDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps') # ARM/Apple Silicon Acceleration
    return torch.device('cpu')

device = get_device()

def train(args):
    # 1. Read Config
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(f"Configuration loaded. Training on {device}")
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # 2. Seeding (Platform Independent)
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # 3. Dataset Setup
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'])
    
    # Batch size is strictly 1 because the custom model expects specific input handling
    train_dataset = DataLoader(voc,
                               batch_size=1,
                               shuffle=True,
                               num_workers=4)
    
    # 4. Model Setup
    faster_rcnn_model = FasterRCNN(model_config,
                                   num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    # 5. Checkpoint Directory
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    # 6. Training Loop
    for i in range(num_epochs):
        rpn_cls_losses = []
        rpn_loc_losses = []
        frcnn_cls_losses = []
        frcnn_loc_losses = []
        
        optimizer.zero_grad()
        
        # Use tqdm for progress bar
        pbar = tqdm(train_dataset, desc=f"Epoch {i+1}/{num_epochs}")
        
        for im, target, fname in pbar:
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            
            # Forward Pass
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            # Loss Calculation
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
            
            # Logging
            rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_cls_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_loc_losses.append(frcnn_output['frcnn_localization_loss'].item())
            
            # Gradient Accumulation
            loss = loss / acc_steps
            loss.backward()
            
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            step_count += 1
            
            # Update Progress Bar
            pbar.set_postfix({'Loss': f"{loss.item()*acc_steps:.4f}"})

        # End of Epoch
        # Ensure any remaining gradients are applied
        if step_count % acc_steps != 1:
            optimizer.step()
            optimizer.zero_grad()
            
        # Save Checkpoint
        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
        torch.save(faster_rcnn_model.state_dict(), ckpt_path)
        
        # Print Epoch Stats
        print(f"\nEpoch {i+1} Summary:")
        print(f"RPN Cls: {np.mean(rpn_cls_losses):.4f} | RPN Loc: {np.mean(rpn_loc_losses):.4f}")
        print(f"FRCNN Cls: {np.mean(frcnn_cls_losses):.4f} | FRCNN Loc: {np.mean(frcnn_loc_losses):.4f}")
        
        scheduler.step()

    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)