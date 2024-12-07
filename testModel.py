# testModel.py

import torch
from models.visual_model import AVmodel

if __name__ == "__main__":
    # Initialize the model
    num_classes = 28
    num_latents = 4
    dim = 768
    model = AVmodel(num_classes=num_classes, num_latents=num_latents, dim=dim)
    model.eval()  # Set to evaluation mode

    # Create dummy inputs
    dummy_pc = torch.randn(1, 3, 224, 224)  # [B, 3, 224, 224]
    dummy_rgb = torch.randn(1, 8, 3, 224, 224)  # [B, no_of_frames, 3, 224, 224]

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_pc, dummy_rgb)
    print(f"Logits shape: {logits.shape}")  # Expected: [1, 28]
