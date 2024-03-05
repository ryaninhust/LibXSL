# LIBXSL: A ML Package for Applying LLMs to Text Classification

LIBXSL is a machine learning package that leverages large language models (LLMs) for text classification tasks. It provides a flexible and scalable framework for training and evaluating text classification models using state-of-the-art LLMs.

## Features

- Easy integration with Hugging Face Transformers
- Support for distributed training with PyTorch
- Customizable loss functions for various classification tasks
- Comprehensive logging and evaluation metrics

## Installation

To install the package, run:

```sh
pip install libxsl
```

### Special Dependency

LIBXSL also depends on a library available from a GitHub repository. This dependency will be automatically installed:

```sh
pip install git+https://github.com/ryaninhust/pyxclib.git
```

## Usage

### Training a Model

To train a model, you need a configuration file (in YAML format) specifying the training parameters, dataset paths, and model configurations. Here's an example configuration file:

```yaml
model_name: "bert-base-uncased"
train_data_file: "path/to/train/data"
test_data_file: "path/to/test/data"
max_length: 128
batch_size: 32
num_epochs: 10
pretrained_lr: 2e-5
label_embedding_lr: 1e-3
pretrained_weight_decay: 0.01
label_embedding_weight_decay: 0.01
positive_weight: 1.0
loss_fn: "LRLR"
omega: 1.0
kernel_approx: true
log_file_path: "training.log"
model_save_path: models/model.pth
prediction_save_path: outputs/predictions.npy
```

Run the training script with the following command:

```python
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

world_size = torch.cuda.device_count()

train_dataset = TextClassificationDataset(config['train_data_file'], tokenizer, config['max_length'])
test_dataset = TextClassificationDataset(config['test_data_file'], tokenizer, config['max_length'], num_classes=train_dataset.num_classes)
mp.spawn(train, args=(world_size, train_dataset, test_dataset, config), nprocs=world_size, join=True)
mp.spawn(predict, args=(world_size, test_dataset, config), nprocs=world_size, join=True)
```

### Customizing the Model and Loss Functions

You can customize the model and loss functions by editing the corresponding files in the package. For example, to add a new loss function, update the `loss_fn.py` file and add the new function to the loss function dictionary.

## Contributing

We welcome contributions to the NEO project! If you have any ideas, bug reports, or improvements, please submit an issue or a pull request on our [GitHub repository](https://github.com/ryaninhust/neo).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

