from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
from pprint import pprint

# My modules
import sys, os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg_path = os.path.join(root_path, "Factory/packages")
sys.path.insert(0, pkg_path)
print(f"Adding to Python path: {pkg_path}")


import customconfig
import data
import models
from metrics.eval_detr_metrics import eval_detr_metrics
from trainer import TrainerDetr
from experiment import ExperimentWrappper

def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    parser.add_argument('--test-only', '-t',  action='store_true', default=False)
    parser.add_argument('--local_rank', default=0)
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config, args

if __name__ == '__main__':
    from pprint import pprint 
    np.set_printoptions(precision=4, suppress=True)
    
    # Get configuration from command line arguments
    config, args = get_values_from_args()
    system_info = customconfig.Properties('./system.json')
    
    # Check if we should use distributed training
    use_distributed = True
    try:
        # Try to initialize distributed training
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        print(f"INFO::{__file__}::Running in distributed mode on rank {rank}.")
        config['trainer']['multiprocess'] = True
        # Set device to current rank
        torch.cuda.set_device(rank)
    except Exception as e:
        # Fall back to non-distributed mode
        print(f"INFO::{__file__}::Running in non-distributed mode. Reason: {e}")
        use_distributed = False
        rank = 0
        config['trainer']['multiprocess'] = False
    
    experiment = ExperimentWrappper(
        config,  # set run id in cofig to resume unfinished run!
        system_info['wandb_username'],
        no_sync=True)  # Disable wandb synchronization
    
    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(system_info['datasets_path'], system_info["sim_root"], config['dataset'], gt_caching=True, feature_caching=False)

    # Disable visualization since we're not using wandb
    trainer = TrainerDetr(
            config['trainer'], experiment, dataset, config['data_split'], 
            with_norm=True, with_visualization=False)  # Disable visualization to avoid wandb directory issues
    trainer.init_randomizer()

    # --- Model ---
    model, criterion = models.build_model(config)
    model_without_ddp = model
    
    if use_distributed:
        # DDP setup for multi-GPU training
        torch.cuda.set_device(rank)
        model.cuda(rank)
        criterion.cuda(rank)
        
        # Wrap model for distributed training
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        device_str = f"cuda:{rank}"
    else:
        # Single GPU or CPU setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion.to(device)
        device_str = str(device)
        
    print(f"Train::Info::Using device: {device_str}")
    
    # Load pre-trained model if available
    if config["NN"]["step-trained"] is not None and os.path.exists(config["NN"]["step-trained"]):
        state_dict = torch.load(config["NN"]["step-trained"], map_location=device_str)["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"Train::Info::Load Pre-step-trained model: {config['NN']['step-trained']}")

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Train::Info::Number of params: {n_parameters}')

    if not args.test_only:    
        trainer.fit(model, model_without_ddp, criterion, rank, config)
    else:
        config["loss"]["lepoch"] = -1
        if config["NN"]["pre-trained"] is None or not os.path.exists(config["NN"]["pre-trained"]):
            print("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    # Only run final evaluation on rank 0 in distributed mode, or always in non-distributed mode
    if not use_distributed or rank == 0:
        try:
            print("Train::Info::Running final evaluation on best model")
            model.load_state_dict(experiment.get_best_model()['model_state_dict'])
            datawrapper = trainer.datawraper
            
            final_metrics = eval_detr_metrics(model, criterion, datawrapper, rank, 'validation')
            experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
            pprint(final_metrics)
            
            final_metrics = eval_detr_metrics(model, criterion, datawrapper, rank, 'test')
            experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
            pprint(final_metrics)
        except Exception as e:
            print(f"Train::Warning::Final evaluation failed: {e}")
            
    # Clean up distributed process group if used
    if use_distributed:
        dist.destroy_process_group()
        
