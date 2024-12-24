import torchvision.transforms as transforms
import torchvision
import os
import torch
from torch.utils.data import DataLoader

from setup.globals import root_directory


def DataProcessorCIFAR(batch_size):
    '''
    Loads the image data and processes it with the dataloader files 
    
    Args:
        batch_size
        
    Returns:
        train_loader: training data
        val_loader: validation data
        test_dataset: testing data
        
    '''
    
    #Tranformations for training data
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    
    # Transforms for validation data (no augmentation)
    transform_test = transforms.Compose(
        [
            transforms.Resize((32,32)), # This image size is currently set to the image size of the cifar10 dataset
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Load the training dataset
    train_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join('.', "data", "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )

    # Wrap training dataset with batchsize and shuffling
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    data_dir = os.path.join(root_directory, "data")

    # Load the testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Wrap testing dataset with batchsize and shuffling for validation
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_dataset