import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import sigmoid
from transformers import AutoModel

from .model import LLMClassifier
from .utils import setup, cleanup, initialize_dataloader, setup_logging, gather_tensor



def predict(rank: int, world_size: int, predict_dataset, config: dict) -> None:
    """
    Perform prediction using a distributed environment.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        predict_dataset: Dataset to be used for prediction.
        config (dict): Configuration dictionary containing model and path settings.
    """

    assert world_size == 1
    try:
        # Setup the distributed environment
        setup(rank, world_size)
        setup_logging(config)

        # Initialize the dataloader
        predict_dataloader = initialize_dataloader(rank, world_size, predict_dataset, config, shuffle=False)

        # Load the pretrained model
        model_name = config['model_name']
        tf_args = {'add_pooling_layer': False} if model_name.startswith('bert-base') else {}
        pretrained_model = AutoModel.from_pretrained(model_name, **tf_args)

        # Initialize the classifier
        model = LLMClassifier(pretrained_model, num_classes=predict_dataset.num_classes).to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model.load_state_dict(torch.load(config['model_save_path']))
        model.eval()

        if rank == 0:
            logging.info("Executeing prediction task")
            predict_dataloader = tqdm(predict_dataloader, desc=f"Prediction", unit="batch", leave=False)

        all_outputs = []
        with torch.no_grad():
            for batch in predict_dataloader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)

                # Forward pass
                predictions = model(input_ids, attention_mask, config['euclidean'], config['kernel_approx'])
                prob_predictions = sigmoid(predictions)
                all_outputs.append(prob_predictions)

            # Concatenate all outputs along the first dimension
            all_outputs = torch.cat(all_outputs, dim=0)

        # Gather all tensors from distributed processes
        all_outputs = gather_tensor(all_outputs, rank, world_size)
        if rank == 0:
            all_outputs = all_outputs[:len(predict_dataset)]
            np.save(config['prediction_save_path'], all_outputs)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        cleanup()
        raise e
    finally:
        # Cleanup the distributed environment
        cleanup()




