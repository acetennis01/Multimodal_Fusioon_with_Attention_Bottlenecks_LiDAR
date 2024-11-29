import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataloader.av_data import KITTIMultiDriveDataset
from models.visual_model import AVmodel

def parse_options():
    parser = argparse.ArgumentParser(description="Multimodal Bottleneck Attention with KITTI Dataset")

    ##### TRAINING DYNAMICS
    parser.add_argument('--gpu_id', type=str, default="cpu", help='the GPU id')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')  # Lower default batch size
    parser.add_argument('--num_epochs', type=int, default=15, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    ##### ADAPTER AND LATENT PARAMETERS
    parser.add_argument('--adapter_dim', type=int, default=8, help='dimension of the low-rank adapter')
    parser.add_argument('--num_latent', type=int, default=4, help='number of latent tokens')
    parser.add_argument('--num_classes', type=int, default=28, help='number of output classes')

    ##### DATA
    parser.add_argument('--data_root', type=str, default='/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_26', help='path to KITTI dataset')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    if opts.gpu_id.lower() == "cpu" or not torch.cuda.is_available():
        opts.device = torch.device("cpu")
    else:
        opts.device = torch.device(f"cuda:{opts.gpu_id}")  # Updated GPU ID handling
    return opts

def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.train()
    scaler = GradScaler()  # Add gradient scaler for mixed precision

    for batch_idx, (point_clouds, rgb_frames, _, oxts_data) in enumerate(train_data_loader):
        print(f"Processing batch {batch_idx + 1}/{len(train_data_loader)}")

        # Move data to device
        point_clouds = point_clouds.to(device)
        rgb_frames = rgb_frames.to(device)
        oxts_data = oxts_data.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            preds = model(point_clouds, rgb_frames)
            labels = oxts_data[:, -1].long()  # Ensure labels are correct
            _loss = loss_fn(preds, labels)
        
        # Backward pass with scaler
        scaler.scale(_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(_loss.item())
        sum_correct_pred += (torch.argmax(preds, dim=1) == labels).sum().item()
        total_samples += len(labels)

        # Log training stats periodically
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_data_loader)} - Loss: {np.mean(epoch_loss):.4f}")
    
    acc = round(sum_correct_pred / total_samples, 5) * 100
    return np.mean(epoch_loss), acc

def val_one_epoch(val_data_loader, model, loss_fn, device):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        for point_clouds, rgb_frames, _, oxts_data in val_data_loader:
            point_clouds = point_clouds.to(device)
            rgb_frames = rgb_frames.to(device)
            oxts_data = oxts_data.to(device)

            preds = model(point_clouds, rgb_frames)
            labels = oxts_data[:, -1].long()
            _loss = loss_fn(preds, labels)

            epoch_loss.append(_loss.item())
            sum_correct_pred += (torch.argmax(preds, dim=1) == labels).sum().item()
            total_samples += len(labels)

    acc = round(sum_correct_pred / total_samples, 5) * 100
    return np.mean(epoch_loss), acc

def collate_fn(batch):
    point_clouds, rgb_frames, timestamps, oxts_data = [], [], [], []
    
    # Find the max number of points in the batch for padding
    max_points = max(pc.shape[0] for pc, _, _, _ in batch)
    
    for point_cloud, rgb_frame, timestamp, oxts in batch:
        # Pad the point cloud to the max size in the batch
        padded_pc = torch.nn.functional.pad(
            point_cloud, (0, 0, 0, max_points - point_cloud.shape[0]), value=0
        )
        point_clouds.append(padded_pc)
        rgb_frames.append(rgb_frame)
        timestamps.append(timestamp)
        oxts_data.append(oxts)
    
    # Stack the padded point clouds and other data
    point_clouds = torch.stack(point_clouds)
    rgb_frames = torch.stack(rgb_frames)
    oxts_data = torch.stack(oxts_data)
    
    return point_clouds, rgb_frames, timestamps, oxts_data

def train_test(args):
    dataset = KITTIMultiDriveDataset(root_dir=args.data_root)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,  # Use the updated collate_fn
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,  # Use the updated collate_fn
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )

    model = AVmodel(num_classes=args.num_classes, num_latents=args.num_latent, dim=args.adapter_dim)
    model.to(args.device)
    print("\t Model Loaded")
    print('\t Trainable params = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []

    print("\t Started Training")
    for epoch in range(args.num_epochs):
        torch.cuda.empty_cache()  # Clear memory before each epoch
        loss, acc = train_one_epoch(trainloader, model, optimizer, loss_fn, args.device)
        val_loss, val_acc = val_one_epoch(valloader, model, loss_fn, args.device)

        print('\nEpoch....', epoch + 1)
        print("Training loss & accuracy......", round(loss, 4), round(acc, 3))
        print("Validation loss & accuracy......", round(val_loss, 4), round(val_acc, 3))
        best_val_acc.append(val_acc)

    print("\n\t Completed Training \n")  
    print("\t Best Results........", np.max(np.asarray(best_val_acc)))

if __name__ == "__main__":
    opts = parse_options()
    train_test(args=opts)

