import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import logging

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cleanup():
    dist.destroy_process_group()

# Function to initialize data loaders
def initialize_dataloader(rank, world_size, dataset, config, shuffle=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler)
    return dataloader

def setup_logging(config):
    logging.basicConfig(filename=config['log_file_path'], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gather_tensor(tensor, rank, world_size):

    if rank == 0:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=gathered_tensors, dst=0)
        gathered_tensors = torch.cat(gathered_tensors)
        gathered_tensors = gathered_tensors.cpu().numpy()
        return gathered_tensors
    else:
        dist.gather(tensor, gather_list=None, dst=0)
        return None


def gather_all_tensor(tensor, rank, world_size):
    """
    Gather tensors from all processes and concatenate them in order.

    Args:
        tensor (torch.Tensor): The tensor to gather.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        numpy.ndarray: The concatenated tensors if rank is 0, None otherwise.
    """

    # Gather tensors from all ranks
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)

    # Concatenate the gathered tensors
    gathered_tensors = torch.cat(gathered_tensors).cpu().numpy()

    return gathered_tensors
