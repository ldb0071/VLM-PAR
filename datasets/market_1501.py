import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoProcessor
from vlmpar.registery import DATASETS

@DATASETS.register('market_1501')
class Market1501(Dataset):
     def __init__(self, root_dir,
                    split="training",
                    processor=None):
          
          self.root_dir = root_dir
          self.processor = AutoProcessor.from_pretrained(processor, use_fast=True)
          self.split = split

          attributes = {
               'gender': ['male', 'female'],
               'hair': ['short hair', 'long hair'],
               'up': ['long sleeve', 'short sleeve'],
               'down': ['long lower body clothing', 'short'],
               'clothes': ['dress', 'pants'],
               'hat': ['no', 'yes'],
               'backpack': ['no', 'yes'],
               'bag': ['no', 'yes'],
               'handbag': ['no', 'yes'],
               'age': ['young', 'teenager', 'adult', 'old'],
               'upblack': ['no', 'yes'],
               'upwhite': ['no', 'yes'],
               'upred': ['no', 'yes'],
               'uppurple': ['no', 'yes'],
               'upyellow': ['no', 'yes'],
               'upgray': ['no', 'yes'],
               'upblue': ['no', 'yes'],
               'upgreen': ['no', 'yes'],
               'downblack': ['no', 'yes'],
               'downwhite': ['no', 'yes'],
               'downpink': ['no', 'yes'],
               'downpurple': ['no', 'yes'],
               'downyellow': ['no', 'yes'],
               'downgray': ['no', 'yes'],
               'downblue': ['no', 'yes'],
               'downgreen': ['no', 'yes'],
               'downbrown': ['no', 'yes']
          }

          if split == "training":
               images_dir = os.path.join(root_dir, "bounding_box_train")
               self.images = [os.path.join(images_dir, img_name) for img_name in os.listdir(images_dir)]
          elif split == "testing":
               images_dir = os.path.join(root_dir, "bounding_box_test")
               self.images = [os.path.join(images_dir, img_name) for img_name in os.listdir(images_dir)]

          # Filter out non-jpg files, junk images, and images with person_id 0
          self.images = [img for img in self.images if img.endswith('.jpg') and 
                        not os.path.basename(img).startswith('-1') and 
                        int(os.path.basename(img).split('_')[0]) != 0]

          # Get all attributes for both train and test
          all_labels = self._get_attributes(
               attributes,
               os.path.join(root_dir, 'market_attribute.mat')
          )

          self.labels = {key: [] for key in attributes.keys()}
     
          self.label_maps = {
               'gender': {
                    'label_to_id': {'male': 0, 'female': 1},
                    'id_to_label': {0: 'male', 1: 'female'}
               },
               'hair': {
                    'label_to_id': {'short hair': 0, 'long hair': 1},
                    'id_to_label': {0: 'short hair', 1: 'long hair'}
               },
               'up': {
                    'label_to_id': {'long sleeve': 0, 'short sleeve': 1},
                    'id_to_label': {0: 'long sleeve', 1: 'short sleeve'}
               },
               'down': {
                    'label_to_id': {'long lower body clothing': 0, 'short': 1},
                    'id_to_label': {0: 'long lower body clothing', 1: 'short'}
               },
               'clothes': {
                    'label_to_id': {'dress': 0, 'pants': 1},
                    'id_to_label': {0: 'dress', 1: 'pants'}
               },
               'hat': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'backpack': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'bag': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'handbag': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'age': {
                    'label_to_id': {'young': 0, 'teenager': 1, 'adult': 2, 'old': 3},
                    'id_to_label': {0: 'young', 1: 'teenager', 2: 'adult', 3: 'old'}
               },
               'upblack': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upwhite': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upred': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'uppurple': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upyellow': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upgray': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upblue': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'upgreen': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downblack': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downwhite': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downpink': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downpurple': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downyellow': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downgray': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downblue': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downgreen': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               },
               'downbrown': {
                    'label_to_id': {'no': 0, 'yes': 1},
                    'id_to_label': {0: 'no', 1: 'yes'}
               }
          }

          for img_path in self.images:
               person_id = int(os.path.basename(img_path).split('_')[0])
               if person_id not in all_labels:
                    continue
               for key in self.labels.keys():
                    self.labels[key].append(self.label_maps[key]['label_to_id'][all_labels[person_id][key]])
          
     def __len__(self):
          return len(self.images)

     def __getitem__(self, idx):
          image = Image.open(self.images[idx])
          if self.processor is not None:
               processed = self.processor(images=image, return_tensors="pt")
               image = processed['pixel_values'].squeeze(0)

          target_labels = {}
          for key, values in self.labels.items():
               raw_label = values[idx]
               target_labels[key] = torch.tensor(raw_label, dtype=torch.long)

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

     def _extract_scalar(self, val):
          while isinstance(val, (np.ndarray, list)) and len(val) > 0:
               val = val[0]
          try:
               return int(val)
          except:
               return None

     def _interpret(self, attributes, attr_name, val):
          v = self._extract_scalar(val)
          if v is None:
               return 'unknown'
          if attr_name == 'age':
               return attributes[attr_name][v - 1] if 1 <= v <= 4 else 'unknown'
          else:
               return attributes[attr_name][v - 1] if 1 <= v <= 2 else 'unknown'

     def _get_attributes(self, attributes, mat_file_path):
          # Load attribute file
          data = loadmat(mat_file_path)
          market_attribute = data['market_attribute']
          attributes_data = market_attribute[0, 0]
          
          # Get both train and test splits
          train_attr = attributes_data['train'][0, 0]
          test_attr = attributes_data['test'][0, 0]
          
          result = {}
          
          # Process training data
          num_train_people = len(train_attr['gender'].squeeze())
          for i in range(num_train_people):
               person_id = i + 1
               person_dict = {}
               for attr_name in attributes:
                    if attr_name not in train_attr.dtype.names:
                         person_dict[attr_name] = 'unknown'
                         continue
                    attr_values = train_attr[attr_name].squeeze()
                    if i >= len(attr_values):
                         person_dict[attr_name] = 'unknown'
                         continue
                    person_dict[attr_name] = self._interpret(attributes, attr_name, attr_values[i])
               result[person_id] = person_dict
               
          # Process testing data
          num_test_people = len(test_attr['gender'].squeeze())
          for i in range(num_test_people):
               person_id = num_train_people + i + 1  # Continue IDs from where train left off
               person_dict = {}
               for attr_name in attributes:
                    if attr_name not in test_attr.dtype.names:
                         person_dict[attr_name] = 'unknown'
                         continue
                    attr_values = test_attr[attr_name].squeeze()
                    if i >= len(attr_values):
                         person_dict[attr_name] = 'unknown'
                         continue
                    person_dict[attr_name] = self._interpret(attributes, attr_name, attr_values[i])
               result[person_id] = person_dict
               
          return result