import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoProcessor
from vlmpar.registery import DATASETS

@DATASETS.register('pa_100k')
class PA100K(Dataset):
     def __init__(self, root_dir,
                    split="training",
                    processor=None):
          
          self.root_dir = root_dir
          self.processor = AutoProcessor.from_pretrained(processor, use_fast=True)
          self.split = split

          if split == "training":
               data = pd.read_csv(os.path.join(root_dir, "train.csv"))
          elif split == "validation":
               data = pd.read_csv(os.path.join(root_dir, "val.csv"))
          elif split == "testing":
               data = pd.read_csv(os.path.join(root_dir, "test.csv"))

          self.images = [os.path.join(root_dir, "data", img_name) for img_name in data["Image"].tolist()]
          
          self.labels = {
               'gender': data["Female"].tolist(),
               'age_over_60': data["AgeOver60"].tolist(),
               'age_18_60': data["Age18-60"].tolist(),
               'age_under_18': data["AgeLess18"].tolist(),
               'view_front': data["Front"].tolist(),
               'view_side': data["Side"].tolist(),
               'view_back': data["Back"].tolist(),
               'hat': data["Hat"].tolist(),
               'glasses': data["Glasses"].tolist(),
               'hand_bag': data["HandBag"].tolist(),
               'shoulder_bag': data["ShoulderBag"].tolist(),
               'backpack': data["Backpack"].tolist(),
               'hold_objects_in_front': data["HoldObjectsInFront"].tolist(),
               'short_sleeve': data["ShortSleeve"].tolist(),
               'long_sleeve': data["LongSleeve"].tolist(),
               'upper_stride': data["UpperStride"].tolist(),
               'upper_logo': data["UpperLogo"].tolist(),
               'upper_plaid': data["UpperPlaid"].tolist(),
               'upper_splice': data["UpperSplice"].tolist(),
               'lower_stripe': data["LowerStripe"].tolist(),
               'lower_pattern': data["LowerPattern"].tolist(),
               'long_coat': data["LongCoat"].tolist(),
               'trousers': data["Trousers"].tolist(),
               'shorts': data["Shorts"].tolist(),
               'skirt_dress': data["Skirt&Dress"].tolist(),
               'boots': data["boots"].tolist(),
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

