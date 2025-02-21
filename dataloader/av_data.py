import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv
from PIL import Image

class KITTIMultiDriveDataset(Dataset):
    def __init__(self, root_dirs, num_images_per_clip=8):

        super(KITTIMultiDriveDataset, self).__init__()

        # Allow a single root directory as well as a list
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.num_images_per_clip = num_images_per_clip

        # Define image transform
        self.visual_transforms = Tv.Compose([
            Tv.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Convert grayscale to RGB
            Tv.Resize((224, 224)),
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize lists to store all file paths and timestamps
        self.point_cloud_paths = []
        self.image_paths = []
        self.oxts_paths = []
        self.timestamps = []

        # Iterate over each provided root directory
        for root_dir in self.root_dirs:
            # Each root_dir is assumed to contain subdirectories (e.g. 2011_09_26, 2011_09_28, etc.)
            for date_folder in sorted(os.listdir(root_dir)):
                date_folder_path = os.path.join(root_dir, date_folder)
                if not os.path.isdir(date_folder_path):
                    continue  # Skip files or non-directories

                # Each date folder should contain drive folders ending with '_sync'
                drives = [os.path.join(date_folder_path, d) 
                          for d in os.listdir(date_folder_path) if d.endswith('_sync')]
                for drive in drives:
                    # Paths for point clouds, images, oxts, and timestamps
                    point_cloud_dir = os.path.join(drive, 'velodyne_points/data')
                    image_dirs = [os.path.join(drive, f'image_0{i}/data') for i in range(4)]
                    oxts_dir = os.path.join(drive, 'oxts/data')
                    timestamps_path = os.path.join(drive, 'velodyne_points/timestamps.txt')

                    # List files and sort to ensure alignment
                    point_cloud_files = sorted(os.listdir(point_cloud_dir))
                    image_files = [sorted(os.listdir(image_dir)) for image_dir in image_dirs]
                    oxts_files = sorted(os.listdir(oxts_dir))

                    # Read the timestamps
                    with open(timestamps_path, 'r') as f:
                        timestamps = [line.strip() for line in f.readlines()]

                    # Append file paths for each sample in the drive
                    for i, point_cloud_file in enumerate(point_cloud_files):
                        self.point_cloud_paths.append(os.path.join(point_cloud_dir, point_cloud_file))
                        self.image_paths.append([
                            os.path.join(image_dir, image_files[cam][i]) 
                            for cam, image_dir in enumerate(image_dirs)
                        ])
                        self.oxts_paths.append(os.path.join(oxts_dir, oxts_files[i]))
                        self.timestamps.append(timestamps[i])

    def __len__(self):
        return len(self.point_cloud_paths)

    def load_oxts_data(self, oxts_path):
        # Load OXTS data from a text file (assuming space-separated values)
        with open(oxts_path, 'r') as f:
            oxts_data = f.readline().strip().split()
        oxts_data = np.array(oxts_data, dtype=np.float32)  # Convert to a float numpy array
        return torch.tensor(oxts_data)

    def __getitem__(self, idx):
        # Load point cloud data (.bin files)
        point_cloud_path = self.point_cloud_paths[idx]
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)  # XYZ + Intensity
        point_cloud = torch.tensor(point_cloud[:, :3], dtype=torch.float32)  # Only XYZ

        # Load images from multiple cameras
        image_paths = self.image_paths[idx]
        images = [Image.open(img_path) for img_path in image_paths]
        transformed_images = [self.visual_transforms(img) for img in images]
        rgb_frames = torch.stack(transformed_images, dim=0)

        # Load OXTS data
        oxts_path = self.oxts_paths[idx]
        oxts_data = self.load_oxts_data(oxts_path)

        # Get timestamp (as a string)
        timestamp = self.timestamps[idx]

        return point_cloud, rgb_frames, timestamp, oxts_data

# Example usage:
# To load data from multiple date folders:
# dataset = KITTIMultiDriveDataset(root_dirs=[
#     '/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_26',
#     '/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_28'
# ])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
