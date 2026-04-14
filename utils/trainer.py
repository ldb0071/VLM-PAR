import os
import json
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from vlmpar.utils.logger import init_logger, logger
from vlmpar.utils.data import collate_fn
from vlmpar.registery import build_dataset, build_model
from vlmpar.datasets import *
from vlmpar.models import *
from vlmpar.utils.misc import model_params
from tqdm import tqdm
import time
import glob
from sklearn.metrics import f1_score
import numpy as np

class Trainer:
     def __init__(self, configs):
          self.configs = configs
          self.setup()

     def setup(self):
          self.setup_project()
          self.setup_dataloaders()
          self.setup_model()
          self.setup_criterion()

     def setup_project(self):
          self.output_dir = self.configs.get('project_dir', 'outputs')
          self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
          self.log_dir = os.path.join(self.output_dir, self.configs.get('logging_dir', 'logs'))
          
          os.makedirs(self.output_dir, exist_ok=True)
          os.makedirs(self.checkpoint_dir, exist_ok=True)
          os.makedirs(self.log_dir, exist_ok=True)

          self.checkpoint_freq = self.configs.get('checkpoint_freq', 1) 
          self.keep_checkpoints = self.configs.get('keep_checkpoints', 3) 

          init_logger(logging_dir=self.log_dir, log_level="INFO")
          self.logger = logger

          accelerator_project_config = ProjectConfiguration(
               project_dir=self.output_dir,
               logging_dir=self.log_dir
          )
          kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

          self.accelerator = Accelerator(
               gradient_accumulation_steps=self.configs.get('gradient_accumulation_steps', 1),
               mixed_precision=self.configs.get('mixed_precision', 'fp16'),
               log_with=self.configs.get('report_to', 'tensorboard'),
               project_config=accelerator_project_config,
               kwargs_handlers=[kwargs],
          )

          self.main_process = self.accelerator.is_main_process
          self.device = self.accelerator.device
          self.dtype = torch.bfloat16 if self.accelerator.mixed_precision == "bf16" else \
                         torch.float16 if self.accelerator.mixed_precision == "fp16" else \
                         torch.float32
          
          if self.main_process:
               print("----------------------------------------------------------------")
               self.logger.info("Project configuration:")
               print("----------------------------------------------------------------")
               self.logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
               self.logger.info(f"Gradient accumulation steps: {self.configs.get('gradient_accumulation_steps', 1)}")
               self.logger.info(f"Training batch size: {self.configs.get('train_batch_size', 32)}")
               self.logger.info(f"Validation batch size: {self.configs.get('valid_batch_size', 32)}")
               self.logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
               self.logger.info(f"Number of workers: {self.configs.get('dataloader_num_workers', 4)}")
               self.logger.info(f"Project directory: {self.output_dir}")
               self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
               self.logger.info(f"Log directory: {self.log_dir}")

               with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                    json.dump(self.configs, f, indent=4)

     def setup_dataloaders(self):
          if self.main_process:
               print("----------------------------------------------------------------")
               self.logger.info("Dataset configuration:")
               print("----------------------------------------------------------------")
          
          train_dataset = build_dataset(self.configs['train_dataset'])
          val_dataset = build_dataset(self.configs['val_dataset'])
          
          self.train_loader = DataLoader(
               train_dataset,
               batch_size=self.configs.get('train_batch_size', 32),
               shuffle=True,
               num_workers=self.configs.get('dataloader_num_workers', 8),
               pin_memory=True,
               collate_fn=collate_fn
          )
          
          self.val_loader = DataLoader(
               val_dataset,
               batch_size=self.configs.get('valid_batch_size', 32),
               shuffle=False,
               num_workers=self.configs.get('dataloader_num_workers', 8),
               pin_memory=True,
               collate_fn=collate_fn
          )

          if self.main_process:
               self.logger.info(f"Training dataset size: {len(train_dataset)}")
               self.logger.info(f"Validation dataset size: {len(val_dataset)}")

     def setup_model(self):
          if self.main_process:
               print("----------------------------------------------------------------")
               self.logger.info("Model configuration:")
               print("----------------------------------------------------------------")
          
          self.model = build_model(self.configs['model']).to(self.device, self.dtype)
          
          optimizer_cfg = self.configs.get('optimizer', {})
          optimizer_type = optimizer_cfg.pop('type', 'Adam')
          self.optimizer = getattr(torch.optim, optimizer_type)(
               self.model.parameters(),
               **optimizer_cfg
          )
          
          scheduler_cfg = self.configs.get('scheduler', {})
          if scheduler_cfg:
               scheduler_type = scheduler_cfg.pop('type', 'StepLR')
               self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
                    self.optimizer,
                    **scheduler_cfg
               )
          else:
               self.scheduler = None

          self.model = self.accelerator.prepare(self.model)
          self.optimizer = self.accelerator.prepare(self.optimizer)

          if self.scheduler:
               self.scheduler = self.accelerator.prepare(self.scheduler)
          self.train_loader, self.val_loader = self.accelerator.prepare(
               self.train_loader, self.val_loader
          )

          if self.main_process:
               self.logger.info(f"Model: {self.configs['model']}")
               self.logger.info(f"Number of parameters: {model_params(self.model)[0]}")
               self.logger.info(f"Number of trainable parameters: {model_params(self.model)[1]}")

     def setup_criterion(self):
          self.criterion = nn.CrossEntropyLoss()

     def save_checkpoint(self, epoch, train_loss, val_loss, val_metrics, is_best=False):
          if not self.main_process:
               return

          checkpoint = {
               'epoch': epoch,
               'model_state_dict': self.model.state_dict(),
               'optimizer_state_dict': self.optimizer.state_dict(),
               'train_loss': train_loss,
               'val_loss': val_loss,
               'val_metrics': val_metrics,
               'config': self.configs
          }

          # Save regular checkpoint
          checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
          torch.save(checkpoint, checkpoint_path)
          self.logger.info(f"Saved checkpoint to {checkpoint_path}")

          # Save best checkpoint
          if is_best:
               best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
               torch.save(checkpoint, best_path)
               self.logger.info(f"Saved best model to {best_path}")

          # Remove old checkpoints if we have too many
          checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')))
          if len(checkpoints) > self.keep_checkpoints:
               for old_checkpoint in checkpoints[:-self.keep_checkpoints]:
                    os.remove(old_checkpoint)
                    self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

     def train(self):
          self.model.train()
          self.optimizer.zero_grad()

          num_epochs = self.configs.get('num_train_epochs', 1)
          if self.main_process:
               epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
          else:
               epoch_pbar = range(num_epochs)

          best_val_loss = float('inf')

          for epoch in epoch_pbar:
               # TRAINING
               if self.main_process:
                    self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                    batch_pbar = tqdm(self.train_loader, desc=f"Epochs {epoch + 1}/{num_epochs}", position=1, leave=False)
               else:
                    batch_pbar = self.train_loader

               epoch_loss = 0
               num_batches = 0
               start_time = time.time()

               for batch in batch_pbar:
                    loss, metrics = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1

                    if self.main_process:
                         batch_pbar.set_postfix({
                              'loss': f'{loss:.4f}',
                              'avg_loss': f'{epoch_loss/num_batches:.4f}'
                         })

               if self.main_process:
                    epoch_time = time.time() - start_time
                    avg_loss = epoch_loss / num_batches
                    self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
                    self.logger.info(f"Average loss: {avg_loss:.4f}")

               # EVALUATION
               self.model.eval()
               val_loss = 0
               val_metrics = {}
               num_val_batches = 0

               if self.main_process:
                    self.logger.info("Starting evaluation...")
                    val_pbar = tqdm(self.val_loader, desc="Validation", position=1, leave=False)
               else:
                    val_pbar = self.val_loader

               with torch.no_grad():
                    for batch in val_pbar:
                         loss, metrics = self.eval_step(batch)
                         val_loss += loss
                         num_val_batches += 1

                         for metric_name, metric_value in metrics.items():
                              if metric_name not in val_metrics:
                                   val_metrics[metric_name] = 0
                              val_metrics[metric_name] += metric_value

                         if self.main_process:
                              val_pbar.set_postfix({
                                   'val_loss': f'{loss:.4f}',
                                   'avg_val_loss': f'{val_loss/num_val_batches:.4f}'
                              })

               if self.main_process:
                    avg_val_loss = val_loss / num_val_batches
                    for metric_name in val_metrics:
                         val_metrics[metric_name] /= num_val_batches

                    avg_val_acc = sum(val_metrics.values()) / len(val_metrics)
                    self.logger.info(f"Validation completed")
                    self.logger.info(f"Average validation loss: {avg_val_loss:.4f}")
                    for metric_name, metric_value in val_metrics.items():
                         self.logger.info(f"Validation {metric_name}: {metric_value:.4f}")
                    self.logger.info(f"Average validation accuracy: {avg_val_acc:.4f}")

                    # Save checkpoint
                    is_best = avg_val_loss < best_val_loss
                    if is_best:
                         best_val_loss = avg_val_loss
                    
                    if (epoch + 1) % self.checkpoint_freq == 0 or is_best:
                         self.save_checkpoint(
                              epoch + 1,
                              avg_loss,
                              avg_val_loss,
                              val_metrics,
                              is_best=is_best
                         )

     def train_step(self, batch):
          self.model.train()
          pixel_values = batch["images"].to(self.device, self.dtype)
          
          outputs = self.model(pixel_values, question_type='all')
          
          total_loss = 0
          valid_samples = 0
          
          for question_type, output in outputs.items():
               logits = output["logits"]
               labels = batch["labels"][question_type].to(self.device)
               
               valid_mask = labels != -1
               
               if not valid_mask.any():
                    continue
                    
               valid_logits = logits[valid_mask]
               valid_labels = labels[valid_mask]
               
               if valid_logits.dim() > 2:
                    valid_logits = valid_logits.view(-1, valid_logits.size(-1))
               if valid_labels.dim() > 1:
                    valid_labels = valid_labels.view(-1)
                    
               loss = self.criterion(valid_logits, valid_labels)
               total_loss += loss * valid_mask.sum()  # Weight by number of valid samples
               valid_samples += valid_mask.sum()
          
          if valid_samples > 0:
               total_loss = total_loss / valid_samples
          else:
               total_loss = torch.tensor(0.0, device=self.device)
          
          self.optimizer.zero_grad()
          self.accelerator.backward(total_loss)
          self.optimizer.step()
          
          # Calculate metrics
          metrics = {}
          for question_type, output in outputs.items():
               logits = output["logits"]
               labels = batch["labels"][question_type].to(self.device)
               
               # Create mask for valid labels
               valid_mask = labels != -1
               
               if valid_mask.any():
                    # Filter out invalid samples
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    # Ensure proper shapes
                    if valid_logits.dim() > 2:
                         valid_logits = valid_logits.view(-1, valid_logits.size(-1))
                    if valid_labels.dim() > 1:
                         valid_labels = valid_labels.view(-1)
                        
                    pred = torch.argmax(valid_logits, dim=-1)
                    # Calculate accuracy
                    metrics[f"{question_type}_acc"] = (pred == valid_labels).float().mean().item()
                    # Calculate F1 score
                    metrics[f"{question_type}_f1"] = f1_score(
                         valid_labels.cpu().numpy(),
                         pred.cpu().numpy(),
                         average='weighted'
                    )
               else:
                    metrics[f"{question_type}_acc"] = 0.0
                    metrics[f"{question_type}_f1"] = 0.0
          
          return total_loss.item(), metrics

     def eval_step(self, batch):
          self.model.eval()
          pixel_values = batch["images"].to(self.device, self.dtype)
          
          outputs = self.model(pixel_values, question_type='all')
          
          total_loss = 0
          valid_samples = 0
          metrics = {}
          
          for question_type, output in outputs.items():
               logits = output["logits"]
               labels = batch["labels"][question_type].to(self.device)
               
               valid_mask = labels != -1
               
               if not valid_mask.any():
                    continue
                    
               valid_logits = logits[valid_mask]
               valid_labels = labels[valid_mask]
               
               if valid_logits.dim() > 2:
                    valid_logits = valid_logits.view(-1, valid_logits.size(-1))
               if valid_labels.dim() > 1:
                    valid_labels = valid_labels.view(-1)
                    
               loss = self.criterion(valid_logits, valid_labels)
               total_loss += loss * valid_mask.sum() 
               valid_samples += valid_mask.sum()
               
               pred = torch.argmax(valid_logits, dim=-1)
               # Calculate accuracy
               metrics[f"{question_type}_acc"] = (pred == valid_labels).float().mean().item()
               # Calculate F1 score
               metrics[f"{question_type}_f1"] = f1_score(
                    valid_labels.cpu().numpy(),
                    pred.cpu().numpy(),
                    average='weighted'
               )
          
          if valid_samples > 0:
               total_loss = total_loss / valid_samples
          else:
               total_loss = torch.tensor(0.0, device=self.device)
          
          return total_loss.item(), metrics

     def cleanup(self):
          self.accelerator.end_training()