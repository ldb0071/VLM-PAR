import torch
from typing import List, Tuple, Dict, Any

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, Any]:
     batch = [item for item in batch if item is not None]
     if not batch:
          return None

     images, target_labels = zip(*batch)
     target_labels = {k: torch.stack([labels[k] for labels in target_labels]) for k in target_labels[0].keys()}

     images = torch.stack(images)
     return {
          'images': images,
          'labels': target_labels
     }
