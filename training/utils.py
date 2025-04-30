"""
utils.py

This file contains the implementation of a few utility functions and classes that are needed
in train_model.py, which is a file that facilitiates the training for the Seemore Vision Language Model.

Author: Lauren Rutledge
Date: April 2025
"""
import base64
import io
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms




class CSVBase64ImageDataset(Dataset):
    """
    This function creates a Custom Pytorch Dataset to load base64-encoded images and text captions from a
    CSV input file (as of now, this is specific to the file in images/inputs.csv)

    CSV Format:
        index, b64string_images, caption

    Args:
        csv_path (str): the path to the CSV input file in THIS repo
        image_size (int): The size to which each image will be resized. Here, we are assuming that the size
            of each image is a square
    """

    def __init__(self, csv_path, image_size):
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # normalizing to [-1, 1] like in CLIP
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Decode the base64 image:
        b64_str = self.data.iloc[idx]['b64string_images']
        caption = self.data.iloc[idx]['caption']

        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image)

        # Now, we want to convert the caption into integer tokens - currently use a dummy
        # target of length 20
        target = torch.zeros(20, dtype=torch.long)

        return image, target