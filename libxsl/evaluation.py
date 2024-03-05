import torch
import torch.distributed as dist
import math
import numpy as np


def precisions(predictions, labels, k):

    # Get the indices that would sort each row in descending order
    sorted_indices = np.argsort(-predictions, axis=1)

    # Create an array of k values
    k_values = np.arange(1, k + 1)

    # Calculate precision for each k
    precision_k = []

    for top_k in k_values:
        # Get the top k indices
        top_k_indices = sorted_indices[:, :top_k]

        # Select the true labels corresponding to the top k predictions
        top_k_labels = np.take_along_axis(labels, top_k_indices, axis=1)

        # Calculate precision@k
        true_positives = np.sum(top_k_labels, axis=1)
        precision = true_positives / top_k
        precision_k.append(np.mean(precision))

    return precision_k


def recalls(predictions, labels, k):

    # Get the indices that would sort each row in descending order
    sorted_indices = np.argsort(-predictions, axis=1)

    # Create an array of k values
    k_values = np.arange(1, k + 1)

    # Calculate recall for each k
    recall_k = []

    for top_k in k_values:
        # Get the top k indices
        top_k_indices = sorted_indices[:, :top_k]

        # Select the true labels corresponding to the top k predictions
        top_k_labels = np.take_along_axis(labels, top_k_indices, axis=1)

        # Calculate recall@k
        true_positives = np.sum(top_k_labels, axis=1)
        total_relevant = np.sum(labels, axis=1)
        recall = true_positives / total_relevant
        recall_k.append(np.mean(recall))

    return recall_k



# Metrics Computation Function
def compute_metrics_batch(predictions, labels, k=5):
    batch_precision_at_k = {}
    batch_recall_at_k = {}


    precision = precisions(predictions, labels, k=k)
    recall = recalls(predictions, labels, k=k)


    for top_k in range(1, k+1):
        # Identify true positive predictions at top_k
        batch_precision_at_k[top_k] = precision[top_k-1] * labels.shape[0]
        batch_recall_at_k[top_k] = recall[top_k-1] * labels.shape[0]

    return batch_precision_at_k, batch_recall_at_k


def evaluate_model(model, test_loader, world_size, gpu_rank, loss_fn, config, k=5):
    model.eval()

    # Initialize total loss and sample count on the GPU
    total_test_loss = torch.tensor(0.0).to(gpu_rank)
    total_test_sample = torch.tensor(0.0).to(gpu_rank)

    # Initialize metrics
    all_precision_at_k = {top_k: torch.tensor(0.0).to(gpu_rank) for top_k in range(1, k+1)}
    all_recall_at_k = {top_k: torch.tensor(0.0).to(gpu_rank) for top_k in range(1, k+1)}

    # Initialize local statistics for min, max, mean, and std
    local_min = float('inf')
    local_max = float('-inf')
    local_sum = 0.0
    local_sq_sum = 0.0
    local_count = 0

    with torch.no_grad():
        for batch in test_loader:
            # Move data to GPU
            input_ids = batch['input_ids'].to(gpu_rank)
            attention_mask = batch['attention_mask'].to(gpu_rank)
            labels = batch['labels'].to(gpu_rank)

            # Forward pass
            outputs = model(input_ids, attention_mask, config['euclidean'], config['kernel_approx'], True)

            # Compute loss
            test_loss = loss_fn(outputs, labels)
            total_test_loss += test_loss.item()

            # Convert to numpy for statistics computation
            predictions = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Update batch metrics
            batch_precision_at_k, batch_recall_at_k = compute_metrics_batch(predictions, labels_np, k)
            total_test_sample += labels.size(0)

            for top_k in range(1, k+1):
                all_precision_at_k[top_k] += batch_precision_at_k[top_k]
                all_recall_at_k[top_k] += batch_recall_at_k[top_k]

            # Update local statistics
            local_min = min(local_min, np.min(predictions))
            local_max = max(local_max, np.max(predictions))
            local_sum += np.sum(predictions)
            local_sq_sum += np.sum(predictions ** 2)
            local_count += predictions.size

    # Aggregate metrics and statistics across all GPUs
    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_test_sample, op=dist.ReduceOp.SUM)
    for top_k in range(1, k+1):
        dist.all_reduce(all_precision_at_k[top_k], op=dist.ReduceOp.SUM)
        dist.all_reduce(all_recall_at_k[top_k], op=dist.ReduceOp.SUM)

    dist.all_reduce(torch.tensor(local_min).to(gpu_rank), op=dist.ReduceOp.MIN)
    dist.all_reduce(torch.tensor(local_max).to(gpu_rank), op=dist.ReduceOp.MAX)
    dist.all_reduce(torch.tensor(local_sum).to(gpu_rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor(local_sq_sum).to(gpu_rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor(local_count).to(gpu_rank), op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        # Finalize the computation of the global metrics
        average_test_loss = total_test_loss / world_size / len(test_loader)
        global_mean = local_sum / local_count
        global_std = math.sqrt(local_sq_sum / local_count - global_mean ** 2)

        # Finalize the computation of precision and NDCG for each K
        for top_k in range(1, k+1):
            all_precision_at_k[top_k] = all_precision_at_k[top_k].item() / total_test_sample.item()
            all_recall_at_k[top_k] = all_recall_at_k[top_k].item() / total_test_sample.cpu().item()

        # Prepare the final metrics dictionary
        precision_at_k = {f'precision@{top_k}': round(all_precision_at_k[top_k]*100, 3) for top_k in range(1, k+1)}
        recall_at_k = {f'recall@{top_k}': round(all_recall_at_k[top_k]*100, 3) for top_k in range(1, k+1)}

        return {
            **precision_at_k,
            **recall_at_k,
            'test_loss': average_test_loss.item(),
        }

    return None

