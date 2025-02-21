import os
import numpy as np
import torch
from torch.utils.data import Dataset  # This was missing
import torchvision.transforms as Tv
from PIL import Image


class KITTIMultiDriveDataset(Dataset):
    def __init__(self, root_dir, num_images_per_clip=8):
        super(KITTIMultiDriveDataset, self).__init__()
        self.root_dir = root_dir
        self.num_images_per_clip = num_images_per_clip

        # Define image transform
        self.visual_transforms = Tv.Compose([
            Tv.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Convert grayscale to RGB
            Tv.Resize((224, 224)),
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Get all the drive directories
        self.drives = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.endswith('_sync')]

        # Store all the point cloud and image paths from multiple drives
        self.point_cloud_paths = []
        self.image_paths = []
        self.oxts_paths = []
        self.timestamps = []

        for drive in self.drives:
            # Path to the point clouds and images
            point_cloud_dir = os.path.join(drive, 'velodyne_points/data')
            image_dirs = [
                os.path.join(drive, f'image_0{i}/data') for i in range(4)  # Adjust to include all cameras
            ]
            oxts_dir = os.path.join(drive, 'oxts/data')
            timestamps_path = os.path.join(drive, 'velodyne_points/timestamps.txt')

            # Load all point clouds and images for this drive
            point_cloud_files = sorted(os.listdir(point_cloud_dir))
            image_files = [sorted(os.listdir(image_dir)) for image_dir in image_dirs]
            oxts_files = sorted(os.listdir(oxts_dir))

            # Read the timestamps as strings
            with open(timestamps_path, 'r') as f:
                timestamps = [line.strip() for line in f.readlines()]

            for i, point_cloud_file in enumerate(point_cloud_files):
                self.point_cloud_paths.append(os.path.join(point_cloud_dir, point_cloud_file))
                self.image_paths.append([os.path.join(image_dir, image_files[cam][i]) for cam, image_dir in enumerate(image_dirs)])
                self.oxts_paths.append(os.path.join(oxts_dir, oxts_files[i]))
                self.timestamps.append(timestamps[i])

    def __len__(self):
        return len(self.point_cloud_paths)

    def load_oxts_data(self, oxts_path):
        # Load OXTS data from text file (assuming space-separated values)
        with open(oxts_path, 'r') as f:
            oxts_data = f.readline().strip().split()
        oxts_data = np.array(oxts_data, dtype=np.float32)  # Convert to a float tensor
        return torch.tensor(oxts_data)

    def __getitem__(self, idx):
        # Load point cloud data (.bin files)
        point_cloud_path = self.point_cloud_paths[idx]
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)  # XYZ + Intensity
        point_cloud = torch.tensor(point_cloud[:, :3], dtype=torch.float32)  # Only XYZ

        # Load corresponding images from multiple cameras
        image_paths = self.image_paths[idx]
        images = [Image.open(img_path) for img_path in image_paths]
        transformed_images = [self.visual_transforms(img) for img in images]

        # Stack images
        rgb_frames = torch.stack(transformed_images, dim=0)

        # Load OXTS data
        oxts_path = self.oxts_paths[idx]
        oxts_data = self.load_oxts_data(oxts_path)

        # Get timestamp (as a string in this case)
        timestamp = self.timestamps[idx]

        return point_cloud, rgb_frames, timestamp, oxts_data

# Example usage
# dataset = KITTIMultiDriveDataset(root_dir='/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_26')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)