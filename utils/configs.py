import argparse
import json
import os
from glob import glob
import importlib.util


def parse_args(training=False):
     parser = argparse.ArgumentParser()

     # model config
     parser.add_argument("config", help="model config file path")

     # ======================================================
     # General
     # ======================================================
     parser.add_argument("--seed", default=None, type=int, help="seed for reproducibility")
     parser.add_argument("--dtype", default=None, type=str, help="data type")
     parser.add_argument(
          "--project-dir",
          type=str,
          default="outputs",
          help="The output directory where the model predictions, checkpoints and logs will be written.",
     )
     parser.add_argument(
          "--logging-dir",
          type=str,
          default="logs",
          help="Directory where logs are stored.",
     )

     # ======================================================
     # Inference
     # ======================================================
     if not training:
          parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
          parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
          parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
          parser.add_argument("--model-ckpt", type=str, default='best_model.pt', help="CKPT file")
          
     # ======================================================
     # Training
     # ======================================================
     else:
          parser.add_argument(
               "--gradient-accumulation-steps",
               type=int,
               default=1,
               help="Number of updates steps to accumulate before performing a backward/update pass.",
          )
          parser.add_argument(
               "--mixed-precision",
               type=str,
               default="fp16",
               help="Choose between fp16 and bf16 (bfloat16).",
          )
          parser.add_argument(
               "--report-to",
               type=str,
               default="tensorboard",
               help='Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
          )
          parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
          parser.add_argument("--num-train-epochs", type=int, default=None)
          parser.add_argument(
               "--max-train-steps",
               type=int,
               default=None,
               help="Total number of training steps to perform. If provided, overrides `--num-train-epochs`.",
          )
          parser.add_argument(
               "--train-batch-size", type=int, default=None, help="Batch size (per device) for the training dataloader."
          )
          parser.add_argument(
               "--valid-batch-size", type=int, default=None, help="Batch size (per device) for the validation dataloader."
          )
          parser.add_argument(
               "--validation-split",
               type=float,
               default=0.25,
               help="Proportion of the dataset to use as the validation set. Default is 0.25 (25%)."
          )
          parser.add_argument(
               "--dataloader-num-workers",
               type=int,
               default=0,
               help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
          )

     return parser.parse_args()


def merge_args(cfg, args, training=False):
     """Merge command line arguments into config."""
     args_dict = vars(args)
     for k, v in args_dict.items():
          if v is not None:
               cfg[k] = v
          elif k not in cfg:
              cfg[k] = v
     return cfg


def read_config(config_path):
    """Read Python config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Convert config to dictionary
    cfg = {}
    for key in dir(config_module):
        if not key.startswith('__'):
            value = getattr(config_module, key)
            if isinstance(value, dict):
                # If the value is a dict, keep it as a separate key
                cfg[key] = value
            else:
                cfg[key] = value
    
    # Ensure project_dir is set
    if 'project_dir' not in cfg:
        cfg['project_dir'] = 'outputs'
    
    return cfg


def parse_configs(training=False):
    """Parse both command line arguments and config file."""
    args = parse_args(training)
    cfg = read_config(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")