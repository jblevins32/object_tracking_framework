import numpy as np
from data_processing.data_downloader import DataDownloader
from data_processing.kitti_dataset import KittiDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
from setup.globals import root_directory


def DataProcessorKitti(batch_size, training_split_percentage=0.8, dataset_percentage=1.0, num_classes=4):

    # Download the data
    dataDownloader = DataDownloader()
    dataDownloader.prepareDataset()

    image_dir = os.path.join(root_directory, "dataset/images/training/image_02")
    label_dir = os.path.join(root_directory, "dataset/labels/training/label_02")

    # Determine desired data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ])

    # Create/process the dataset
    dataset = KittiDataset(image_dir=image_dir, label_dir=label_dir, transform=transform, num_classes=num_classes)

    # Split indices for train, validation, and test
    dataset_cutoff_idx = int(len(dataset) * dataset_percentage)
    total_indices = list(range(dataset_cutoff_idx))
    train_indices, val_test_indices = train_test_split(total_indices,
                                                       test_size=1.0-training_split_percentage,
                                                       random_state=42)
    val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)

    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    return train_loader, val_loader, test_dataset

