import os
import sys
import logging
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

from .evaluation import evaluate_model
from .model import LLMClassifier
from .loss_fn import (
    LRSQLoss, LRL2Loss, LRExpLoss, ExpLoss, 
    L2SQLoss, LogSoftmaxLoss, SQLoss
)
from .data import TextClassificationDataset
from .utils import setup, cleanup, initialize_dataloader, setup_logging

# DDP Setup and Cleanup Functions
# Function to initialize model and optimizer
def initialize_model_optimizer(rank, train_dataset, config):
    model_name = config['model_name']
    tf_args = {'add_pooling_layer': False} if model_name.startswith('bert-base') else {}
    pretrained_model = AutoModel.from_pretrained(model_name, **tf_args)
    model = LLMClassifier(pretrained_model, num_classes=train_dataset.num_classes).to(rank)
    
    if not config['train_map_weight']:
        if rank == 0:
            logging.info('disable map_weight training')
        model.map_weight.requires_grad = False

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer_grouped_parameters = [
        {'params': model.module.pretrained_model.parameters(), 'lr': config['pretrained_lr'], 'weight_decay': config['pretrained_weight_decay']},
        {'params': [model.module.weight, model.module.bias], 'lr': config['label_embedding_lr'], 'weight_decay': config['label_embedding_weight_decay']},
        {'params': [model.module.map_weight, model.module.map_bias], 'lr': config['label_embedding_lr'], 'weight_decay': config['label_embedding_weight_decay']}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return model, optimizer

# Function to initialize data loaders
def initialize_dataloaders(rank, world_size, train_dataset, test_dataset, config):
    train_loader = initialize_dataloader(rank, world_size, train_dataset, config, shuffle=True)
    test_loader = initialize_dataloader(rank, world_size, test_dataset, config, shuffle=False)
    return train_loader, test_loader

# Function to select the loss function
def select_loss_function(config, rank, train_dataset):
    pos_weights = torch.full((train_dataset.num_classes,), config['positive_weight']).to(rank)
    
    loss_fn_dict = {
        'LRSQ': LRSQLoss(float(config['omega'])),
        'LRLR': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
        'LRL2': LRL2Loss(float(config['omega'])),
        'LRExp': LRExpLoss(float(config['omega'])),
        'SQ': SQLoss(config['omega']),
        'Exp': ExpLoss(float(config['omega'])),
        'L2SQL': L2SQLoss(float(config['omega'])),
        'LogSoftmax': LogSoftmaxLoss(float(config['omega']), config['kernel_approx'])
    }
    
    if config['loss_fn'] not in loss_fn_dict:
        raise Exception('error loss')

    return loss_fn_dict[config['loss_fn']]



def train(rank: int, world_size: int, train_dataset, test_dataset, config: dict) -> None:
    """
    Train a model using a distributed environment.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        train_dataset: Dataset to be used for training.
        test_dataset: Dataset to be used for evaluation.
        config (dict): Configuration dictionary containing model and training settings.
    """
    try:
        # Setup the distributed environment
        setup(rank, world_size)
        setup_logging(config)

        # Initialize model and optimizer
        model, optimizer = initialize_model_optimizer(rank, train_dataset, config)

        # Initialize data loaders
        train_loader, test_loader = initialize_dataloaders(rank, world_size, train_dataset, test_dataset, config)

        # Select loss function
        loss_fn = select_loss_function(config, rank, train_dataset)

        num_epochs = int(config['num_epochs'])
        num_training_steps = int(len(train_dataset) / config['batch_size'] / world_size * num_epochs)
        num_warmup_steps = int(num_training_steps * config['warmup'])

        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(num_epochs):
            total_loss = torch.tensor(0.0).to(rank)
            model.train()

            if rank == 0:
                train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

            for batch in train_loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                labels = batch['labels'].to(rank)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, config['euclidean'], config['kernel_approx'])
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Aggregate loss across all processes
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                average_loss = total_loss.item() / world_size / len(train_loader)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {average_loss}")

            if (epoch + 1) % 10 == 0:
                # Evaluate model on test dataset
                test_metrics = evaluate_model(model, test_loader, world_size, rank, loss_fn, config)
                if rank == 0:
                    logging.info(f"[Test] Epoch {epoch+1}: {test_metrics}")

                # Evaluate model on training dataset
                train_metrics = evaluate_model(model, train_loader, world_size, rank, loss_fn, config)
                if rank == 0:
                    logging.info(f"[Training] Epoch {epoch+1}: {train_metrics}")

        if rank == 0:
            # Save the trained model
            torch.save(model.state_dict(), config['model_save_path'])

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        cleanup()
        raise e
    finally:
        # Cleanup the distributed environment
        cleanup()

