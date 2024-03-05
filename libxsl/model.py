import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class LLMClassifier(nn.Module):
    def __init__(self):
        super(LLMClassifier, self).__init__()

    def __init__(self, pretrained_model, num_classes, expansion_scale=24):
        super(LLMClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        hidden_size = pretrained_model.config.hidden_size


        embedding_size = 64

        self.reduce_size_layer = nn.Linear(hidden_size, embedding_size)

        # Initialize classification layer weights and biases
        self.init_classification_layer(embedding_size, num_classes)
        
        # Initialize kernel approximation parameters if enabled
        self.init_kernel_approximation(embedding_size, expansion_scale)

    def init_classification_layer(self, hidden_size, num_classes):
        """Initializes the classification layer's parameters."""
        self.weight = nn.Parameter(torch.Tensor(num_classes, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.dropout = nn.Dropout(0.5)

    def init_kernel_approximation(self, hidden_size, expansion_scale):
        """Initializes parameters for the kernel approximation."""
        expanded_hidden_size = hidden_size * expansion_scale
        self.map_weight = nn.Parameter(torch.Tensor(expanded_hidden_size, hidden_size))
        self.map_bias = nn.Parameter(torch.Tensor(expanded_hidden_size))
        #nn.init.xavier_uniform_(self.map_weight)

        nn.init.normal_(self.map_weight, mean=0, std=(1/1)**0.5)

        nn.init.zeros_(self.map_bias)

    def mean_pooling(self, model_output, attention_mask):
        """Performs mean pooling on the model's output."""
        token_embeddings = model_output[0]  # Token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_embedding(self, model_output, attention_mask=None):
        """Extracts the CLS token embedding as sentence representation."""
        return model_output['last_hidden_state'][:, 0]

    def forward(self, input_ids, attention_mask, euclidean=False, kernel_approx=False, inference=False):
        """Defines the forward pass of the classifier."""
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = self.cls_embedding(outputs, attention_mask)
        pooled_output = self.reduce_size_layer(cls_output)

        
        if kernel_approx:
            return self.kernel_approx_logits(pooled_output, inference)
        
        return self.euclidean_or_linear_logits(pooled_output, euclidean)

    def kernel_approx_logits(self, pooled_output, inference):
        """Computes logits using kernel approximation."""
        norm_pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        norm_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)

        #if inference:
        #    return torch.matmul(norm_pooled_output, norm_weight.t())

        #norm_map_weight = torch.nn.functional.normalize(self.map_weight, p=2, dim=1)
        norm_map_weight = self.map_weight

        gamma = 1

        projected_output = torch.exp(torch.matmul(norm_pooled_output, norm_map_weight.t())*gamma/2 - gamma/2)
        projected_weight = torch.exp(torch.matmul(norm_weight, norm_map_weight.t())*gamma/2 - gamma/2)

        return torch.matmul(projected_output, projected_weight.t())

    def euclidean_or_linear_logits(self, pooled_output, euclidean):
        """Computes logits using either Euclidean distance or linear transformation."""
        if euclidean:
            return self.euclidean_logits(pooled_output)
        else:
            return torch.matmul(pooled_output, self.weight.t()) + self.bias

    def euclidean_logits(self, pooled_output):
        """Computes logits based on Euclidean distance."""
        norm_pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        norm_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        pooled_output_sq_sum = torch.sum(norm_pooled_output ** 2, dim=1, keepdim=True)
        weight_sq_sum = torch.sum(norm_weight ** 2, dim=1, keepdim=True)
        logits = torch.matmul(norm_pooled_output, norm_weight.t())
        return logits - weight_sq_sum.t() - pooled_output_sq_sum + 2

