import torch
from av_data import KITTIMultiDriveDataset  # Ensure you import the correct module where KITTIMultiDriveDataset is located
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    point_clouds, rgb_frames, timestamps, oxts_data = zip(*batch)
    
    # Pad the point clouds to have the same number of points
    point_clouds = [torch.tensor(pc) for pc in point_clouds]
    padded_point_clouds = pad_sequence(point_clouds, batch_first=True)  # Pads along the first dimension
    
    # Stack images and OXTS data (assuming they are of fixed size)
    rgb_frames = torch.stack(rgb_frames)  # Stack image frames
    oxts_data = torch.stack(oxts_data)  # Stack OXTS data
    
    return padded_point_clouds, rgb_frames, timestamps, oxts_data

# Path to the KITTI root directory
root_dir = '/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_26'

# Instantiate the dataset
dataset = KITTIMultiDriveDataset(root_dir=root_dir)

# Create a dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Iterate through the dataloader and print the shape of the outputs
for i, (point_cloud, rgb_frames, timestamp, oxts_data) in enumerate(dataloader):
    print(f'Batch {i+1}:')
    print(f'Point Cloud Shape: {point_cloud.shape}')  # Expecting (batch_size, num_points, 3)
    print(f'RGB Frames Shape: {rgb_frames.shape}')  # Expecting (batch_size, num_cameras, H, W, C)
    print(f'Timestamps: {timestamp}')
    print(f'OXTS Data Shape: {oxts_data.shape}')  # Expecting (batch_size, num_oxts_values)

    if i == 2:  # Limit the test to 3 batches
        break
