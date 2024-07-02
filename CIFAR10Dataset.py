import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as tt
import numpy as np
import os
import contextlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class CIFAR10Subset:
    def __init__(self, root='./data', train_percentage=1, batch_size=256, download=True, seed=None):
        self.root = root
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.download = download
        self.seed = seed
        
        # Set the random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Define transformations
        stats = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617))
        self.transform = tt.Compose([
            tt.ToTensor(),
        ])
        
        self.test_transform = tt.Compose([
            tt.ToTensor(),
        ])
        
        # Suppress the "Download files verified" print statement
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            # Load full datasets
            self.full_train_data = datasets.CIFAR10(root=self.root, train=True, download=self.download, transform=self.transform)
            self.full_test_data = datasets.CIFAR10(root=self.root, train=False, download=self.download, transform=self.test_transform)
        
        # Get a subset of the datasets
        self.train_data = self.get_subset(self.full_train_data)
        self.test_data = self.get_subset(self.full_test_data)
        
        # Create DataLoaders
        num_workers = os.cpu_count() - 1
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
    
    def get_subset(self, dataset):
        num_samples = int(len(dataset) * self.train_percentage)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        return subset



    
  