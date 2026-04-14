import torch.nn as nn

def model_params(model: nn.Module):
     """
     Log the number of parameters in a model.

     Args:
          model (torch.nn.Module): The model.
     """
     num_params = 0
     num_params_trainable = 0
     for p in model.parameters():
          num_params += p.numel()
          if p.requires_grad:
               num_params_trainable += p.numel()
     return num_params, num_params_trainable