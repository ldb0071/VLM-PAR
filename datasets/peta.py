import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoProcessor
from vlmpar.registery import DATASETS

@DATASETS.register('peta')
class PETA(Dataset):
     def __init__(self, root_dir,
                    split="training",
                    processor=None):
          
          self.root_dir = root_dir
          self.processor = AutoProcessor.from_pretrained(processor, use_fast=True)
          self.split = split

          # Load PETA.mat file
          data = loadmat(os.path.join(root_dir, "PETA.mat"))
          
          # Define the 35 attributes in exact order
          self.attribute_names = [
               'accessoryHat', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'hairLong',
               'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid',
               'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck',
               'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt',
               'lowerBodyTrousers', 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker',
               'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags',
               'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'personalMale'
          ]
          
          # Get the original attribute names from the data
          original_attributes = [str(name[0][0]) for name in data['peta'][0][0][1]]
          self.attribute_names = original_attributes

          # Create mapping from original attributes to our selected attributes
          self.attribute_indices = [original_attributes.index(attr) for attr in self.attribute_names]
          
          attributes_data = data['peta'][0][0][0]
          split_indices = data['peta'][0][0][3]
          
          if split == "training":
               indices = split_indices[0][0][0][0][0][:, 0] - 1
          elif split == "validation":
               indices = split_indices[0][0][0][0][1][:, 0] - 1
          elif split == "testing":
               indices = split_indices[0][0][0][0][2][:, 0] - 1
          else:
               raise ValueError(f"Invalid split: {split}")
          
          self.images = []
          self.labels = defaultdict(list)
          
          for idx in indices:
               img_name = f"{int(idx+1):05d}.png"
               self.images.append(os.path.join(root_dir, "images", img_name))
               attributes = attributes_data[idx, 4:]
               # Only use the selected attributes
               for attr_idx, attr_name in enumerate(self.attribute_names):
                    self.labels[attr_name].append(int(attributes[self.attribute_indices[attr_idx]]))
          
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
