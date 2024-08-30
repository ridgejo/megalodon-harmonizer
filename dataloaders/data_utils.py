from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler, Sampler
import math

DATA_PATH = Path("/data/engs-pnpl/lina4368")


def get_key_from_batch_identifier(batch_identifier: dict) -> str:
    identifier = {k: batch_identifier[k][0] for k in batch_identifier.keys()}
    return get_key_from_identifier(identifier)


def get_key_from_identifier(identifier: dict) -> str:
    key = f"dat={identifier['dataset']}"
    if "subject" in identifier:
        key += f"_sub={identifier['subject']}"
    return key

def get_dset_encoding(dataset):
    if dataset == "armeni2022":
        return 0
    elif dataset == "gwilliams2022":
        return 1
    elif dataset == "schoffelen2019":
        return 2
    elif dataset == "shafto2014":
        return 3
    else:
        raise ValueError("Dataset not supported")
    
class ComboLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iterators = [iter(dataloader) for dataloader in dataloaders]
        self.num_batches = min(len(dataloader) for dataloader in dataloaders)

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count >= self.num_batches:
            raise StopIteration

        batches = []
        for iterator in self.iterators:
            try:
                batch = next(iterator)
            except StopIteration:
                batch = None
            batches.append(batch)

        self.batch_count += 1
        return tuple(batches)

    def __len__(self):
        return self.num_batches
    
def get_oversampler(dataset, target_size):
    # Calculate the number of samples needed to match the target size
    num_samples = len(dataset)
    oversample_factor = target_size // num_samples + 1

    # Create list of oversampled indices 
    indices = torch.arange(num_samples).repeat(oversample_factor).tolist()
    
    # Trim to target size
    indices = indices[:target_size]

    # Create sampler with indices
    sampler = WeightedRandomSampler(indices, len(indices), replacement=True)
    return sampler

class Oversampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        
        # Calculate how many batches we need and the total number of samples
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.total_samples = self.num_batches * self.batch_size
        
    def __iter__(self):
        # Generate indices with replacement if necessary
        indices = torch.randint(0, self.num_samples, (self.total_samples,))
        
        # # Yield batches of indices
        # for i in range(0, len(indices), self.batch_size):
        #     yield indices[i:i + self.batch_size].tolist()

        # Yield one index at a time
        for i in range(0, len(indices)):
            yield indices[i].item()

    def __len__(self):
        return self.total_samples
    
def get_age_distribution_labels(ages, age_range=(18, 89), sigma=10):
    """
    Get softmax outputs for a batch of ages, supporting float ages.
    
    Parameters:
    ages (list or Tensor): List or Tensor of true ages (can be floats).
    age_range (tuple): The minimum and maximum age for the age bins.
    sigma (float): The standard deviation for the normal distribution.
    
    Returns:
    Tensor: A tensor containing the softmax outputs for each age in the batch.
    """
    
    # Define the age bins based on the age range, keeping them as floats
    age_bins = torch.arange(age_range[0], age_range[1] + 1, step=1).float().requires_grad_(True).to("cpu")
    
    # Convert the list of ages to a tensor
    ages = torch.tensor(ages).float().view(-1, 1).requires_grad_(True).to("cpu")
    
    # Calculate the normal distribution (Gaussian) values for each age in the batch
    gaussian_outputs = torch.exp(-((age_bins - ages)**2) / (2 * sigma**2))
    
    # # Normalize the Gaussian outputs into a softmax distribution
    # softmax_outputs = F.softmax(gaussian_outputs, dim=1)
    
    return gaussian_outputs
