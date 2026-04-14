import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoProcessor
from vlmpar.registery import DATASETS

@DATASETS.register('mivia_par_kd_2025')
class MiviaParKd2025(Dataset):
     def __init__(self, root_dir,
                    split="training",
                    processor=None):
          self.root_dir = root_dir
          self.processor = AutoProcessor.from_pretrained(processor, use_fast=True)
          self.split = split

          data = pd.read_csv(os.path.join(root_dir, split + "_set.txt"), header=None)


          self.images = [os.path.join(root_dir, split + "_set", img_name) for img_name in data[0].tolist()]
          
          self.labels = {
               'upper_body_color': data[1].tolist(),
               'lower_body_color': data[2].tolist(),
               'gender': data[3].tolist(),
               'bag': data[4].tolist(),
               'hat': data[5].tolist()
          }
          
          color_labels = {
               1: 'black',
               2: 'blue',
               3: 'brown',
               4: 'gray',
               5: 'green',
               6: 'orange',
               7: 'pink',
               8: 'purple',
               9: 'red',
               10: 'white',
               11: 'yellow'
          }
          
          self.label_maps = {
               'upper_body_color': {
                    'label_to_id': {str(k): k-1 for k in color_labels.keys()},  # Convert to 0-based indexing
                    'id_to_label': {k-1: v for k, v in color_labels.items()}
               },
               'lower_body_color': {
                    'label_to_id': {str(k): k-1 for k in color_labels.keys()},  # Convert to 0-based indexing
                    'id_to_label': {k-1: v for k, v in color_labels.items()}
               },
               'gender': {
                    'label_to_id': {'0': 0, '1': 1},
                    'id_to_label': {0: 'male', 1: 'female'}
               },
               'bag': {
                    'label_to_id': {'0': 0, '1': 1},
                    'id_to_label': {0: 'no_bag', 1: 'has_bag'}
               },
               'hat': {
                    'label_to_id': {'0': 0, '1': 1},
                    'id_to_label': {0: 'no_hat', 1: 'has_hat'}
               }
          }
          

     def __len__(self):
          return len(self.images)
      
     def __getitem__(self, idx):
          image = Image.open(self.images[idx])
          if self.processor is not None:
               processed = self.processor(images=image, return_tensors="pt")
               image = processed['pixel_values'].squeeze(0)
          
          target_labels = {}
          for key, values in self.labels.items():
               raw_label = str(values[idx])
               if raw_label == '-1':
                    target_labels[key] = torch.tensor(-1, dtype=torch.long)
               else:
                    label_id = self.label_maps[key]['label_to_id'][raw_label]
                    target_labels[key] = torch.tensor(label_id, dtype=torch.long)

          return image, target_labels

     @staticmethod
     def collate_fn(batch):
          batch = [item for item in batch if item is not None]
          if not batch: return None
          pixel_values = torch.stack([item["pixel_values"] for item in batch])
          labels = defaultdict(list)
          label_keys = batch[0]["labels"].keys()
          for item in batch:
               for key in label_keys: labels[key].append(item["labels"][key])
          stacked_labels = {key: torch.stack(label_list) for key, label_list in labels.items()}
          return {"pixel_values": pixel_values, "labels": stacked_labels}
