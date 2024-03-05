import logging
import torch
import numpy as np
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length, num_classes=None, label_map={}):
        self.examples = []
        self.label_map = label_map
        self.max_label_id = -1

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('<TAB>')
                id_, labels, text = parts[-3], [self.get_label_id(i) for i in  parts[-2].split(' ')], parts[-1]

                if num_classes is not None:
                    labels = [label for label in labels if label < num_classes]
                else:
                    self.max_label_id = max(self.max_label_id, max(labels, default=-1))

                self.examples.append((id_, labels, text))

        self.num_classes = num_classes if num_classes is not None else self.max_label_id + 1
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess()
        logging.info(self.label_map)

    def get_label_id(self, label_token):
        if label_token not in self.label_map:
            self.label_map[label_token] = len(self.label_map)
        return self.label_map[label_token]


    def inverse_label_map(self):
        self.inv_label_map = {}
        for key, value in self.label_map.items():
            self.inv_label_map[value] = key


    def get_label_token(self, label_id):
        return self.inv_label_map[label_id]

    def create_one_hot(self, labels):
        label_vector = torch.zeros(self.num_classes)
        for label in labels:
            label_vector[label] = 1
        return label_vector

    def preprocess(self):
        self.ids, self.labels, self.texts = [], [], []
        for idx in range(len(self.examples)):
            id_, labels, text = self.examples[idx]
            self.ids.append(id_)
            self.labels.append(labels)
            self.texts.append(text)

        inputs = self.tokenizer(
            self.texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.input_ids = inputs['input_ids']
        self.attention_masks = inputs['attention_mask']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        label_vector = self.create_one_hot(self.labels[idx])
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_vector}

    def predict_labels(self, predictions, k=1, threshold=0):

        assert predictions.shape[0] == len(self.examples)
        assert predictions.shape[1] == len(self.label_map)

        self.inverse_label_map()
        # Mask out the values below the threshold
        masked_predictions = np.where(predictions > threshold, predictions, -np.inf)

        # Get the indices of the sorted values in descending order
        sorted_indices = np.argsort(masked_predictions, axis=1)[:, ::-1]

        # Select the top-k indices
        top_k_indices = sorted_indices[:, :k]

        # Filter out -inf indices (which were below the threshold)
        result = []
        for i in range(predictions.shape[0]):
            row_indices = top_k_indices[i]
            valid_indices = row_indices[masked_predictions[i, row_indices] != -np.inf]
            valid_labels = [self.get_label_token(idx) for idx in valid_indices]
            result.append([self.ids[i], ' '.join(valid_labels)])
        return result

