import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv
from PIL import Image

class KITTIMultiDriveDataset(Dataset):
    def __init__(self, root_dirs, num_images_per_clip=8):
        """
        Args:
            root_dirs (str or list): A single directory path or a list of directory paths.
                                     Each path can either be a parent folder that contains date folders
                                     or a date folder directly containing drive directories (ending with '_sync').
            num_images_per_clip (int): Number of images per clip (if needed for further processing).
        """
        super(KITTIMultiDriveDataset, self).__init__()

        # Allow a single root directory as well as a list
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.num_images_per_clip = num_images_per_clip

        self.visual_transforms = Tv.Compose([
            Tv.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            Tv.Resize((224, 224)),
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.point_cloud_paths = []
        self.image_paths = []
        self.oxts_paths = []
        self.timestamps = []

        # Iterate over each provided directory
        for root_dir in self.root_dirs:
            # Check if this directory is a date folder by looking for a known subdirectory
            if os.path.exists(os.path.join(root_dir, 'velodyne_points')):
                date_folders = [root_dir]  # Treat root_dir as the date folder
            else:
                # Otherwise, assume it contains multiple date folders
                date_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))]

            for date_folder in date_folders:
                # Look for drive folders ending with '_sync'
                drives = [os.path.join(date_folder, d) for d in os.listdir(date_folder)
                          if d.endswith('_sync') and os.path.isdir(os.path.join(date_folder, d))]
                for drive in drives:
                    point_cloud_dir = os.path.join(drive, 'velodyne_points/data')
                    image_dirs = [os.path.join(drive, f'image_0{i}/data') for i in range(4)]
                    oxts_dir = os.path.join(drive, 'oxts/data')
                    timestamps_path = os.path.join(drive, 'velodyne_points/timestamps.txt')

                    # Make sure the required directories and files exist
                    if not os.path.exists(point_cloud_dir) or not os.path.exists(timestamps_path):
                        continue

                    # List files and sort them
                    point_cloud_files = sorted(os.listdir(point_cloud_dir))
                    image_files = [sorted(os.listdir(image_dir)) for image_dir in image_dirs if os.path.exists(image_dir)]
                    oxts_files = sorted(os.listdir(oxts_dir)) if os.path.exists(oxts_dir) else []

                    # Read timestamps
                    try:
                        with open(timestamps_path, 'r') as f:
                            timestamps = [line.strip() for line in f.readlines()]
                    except Exception as e:
                        print(f"Error reading timestamps from {timestamps_path}: {e}")
                        continue

                    # Check if counts align
                    if len(point_cloud_files) != len(timestamps):
                        print(f"Warning: number of point cloud files and timestamps do not match in {drive}")
                        continue

                    for i, _ in enumerate(point_cloud_files):
                        self.point_cloud_paths.append(os.path.join(point_cloud_dir, point_cloud_files[i]))
                        # Ensure that each camera has the same number of images
                        if all(len(image_list) > i for image_list in image_files):
                            self.image_paths.append([os.path.join(image_dirs[cam], image_files[cam][i])
                                                     for cam in range(len(image_dirs))])
                        else:
                            self.image_paths.append([])  # Or handle the error as needed
                        if oxts_files:
                            self.oxts_paths.append(os.path.join(oxts_dir, oxts_files[i]))
                        else:
                            self.oxts_paths.append("")
                        self.timestamps.append(timestamps[i])

        # Print dataset length for debugging
        print(f"Dataset initialized with {len(self.point_cloud_paths)} samples.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def load_oxts_data(self, oxts_path):
        # Load OXTS data (assuming a space-separated text file)
        if not oxts_path or not os.path.exists(oxts_path):
            return torch.tensor([])  # Or handle missing data as needed
        with open(oxts_path, 'r') as f:
            oxts_data = f.readline().strip().split()
        oxts_data = np.array(oxts_data, dtype=np.float32)
        return torch.tensor(oxts_data)

    def __getitem__(self, idx):
        # Load point cloud data (.bin files)
        point_cloud_path = self.point_cloud_paths[idx]
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        point_cloud = torch.tensor(point_cloud[:, :3], dtype=torch.float32)

        # Load images from multiple cameras
        image_paths = self.image_paths[idx]
        images = [Image.open(img_path) for img_path in image_paths] if image_paths else []
        transformed_images = [self.visual_transforms(img) for img in images]
        rgb_frames = torch.stack(transformed_images, dim=0) if transformed_images else torch.tensor([])

        # Load OXTS data
        oxts_path = self.oxts_paths[idx]
        oxts_data = self.load_oxts_data(oxts_path)

        # Get timestamp
        timestamp = self.timestamps[idx]

        return point_cloud, rgb_frames, timestamp, oxts_data
