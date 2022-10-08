import argparse
import sys
import time
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, List

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Layer Probe")

        # TASK
        parser.add_argument("--embedding", default='clip', type=str,
                            choices=["clip", "resnet50"],
                            help='What type of embeddings per image'
                            )
        # Data
        parser.add_argument("--data_dir", default='../Datasets/cars', help="path to dataset ROOT directory")
        parser.add_argument("-b", "--batch_size", default=32, type=int, metavar="N", help="mini-batch size.")
        parser.add_argument("--num_workers", default=6, type=int, help="Number of workers for dataloader")
        
        # Checkpointing and visulaization
        parser.add_argument('--save_root', default="./outputs/", type=str,help='root directory for saving checkpoints')
        parser.add_argument('--vis_interval', default=100, type=int,help='root directory where models checkpoints exist')
        parser.add_argument('--save_interval', default=500, type=int,help='root directory where models checkpoints exist')

        # wandb
        parser.add_argument(
            "--no_wandb", type=int, default=0, help="no wandb"
        )
        parser.add_argument(
            "--wandb_project_name", type=str, default="",
            help="Wandb sweep may overwrite this!",
        )
        parser.add_argument("--wandb_experiment_name", type=str, default="")
        parser.add_argument("--wandb_entity", type=str)
        parser.add_argument("--wandb_offline", type=str, default="online")
        
                
        # debugging
        parser.add_argument(
            "--debug",
            default=0,
            type = int,
            help="Run the experiment with only a few batches for all"
            "datasets, to ensure code runs without crashing.",
        )
        parser.add_argument(
            "--debug_batches",
            type=int,
            default=6,
            help="Number of batches to run in debug mode.",
        )
        parser.add_argument(
            "--debug_epochs",
            type=int,
            default=3,
            help="Number of epochs to run in debug mode. "
            "If a non-positive number is passed, all epochs are run.",
        )
        
        # optimization
        parser.add_argument("--optimizer", default='AdamW', type=str, help="Number of workers for dataloader")
        parser.add_argument("--lr", default=1e-3, type=float, help="Number of workers for dataloader")
        parser.add_argument("--momentum", default=0.9, type=float, help="Number of workers for dataloader")
        parser.add_argument("--weight_decay", default=1e-4, type=float, help="Number of workers for dataloader")
        parser.add_argument("--epochs", default=10000, type=float, help="Number of workers for dataloader")
        
        # transformation hyperparams
        parser.add_argument("--crop_padding", default=40, type=int, help="padding for randomcrop")
        
        # Relpose
        parser.add_argument("--sampling_mode", default='equivolumetric', type=str, help="Sampling method for Relpose")
        parser.add_argument("--recursion_level", default=3, type=int, help="recursion level for Relpose")
        parser.add_argument("--freeze_encoder", default=0, type=int, help="Freeze encoder in Relpose?")
        
        self.parser = parser
        
    def parse(self, args):
        self.cfg = self.parser.parse_args(args)
        now = datetime.now()
        
        save_root = Path(self.cfg.save_root) / now.strftime("%m-%d-%Y_%H-%M-%S")
        
        self.cfg.save_root = save_root
        self.cfg.ckpt_path = save_root / self.cfg.embedding / 'weights'
        self.cfg.vis_path = save_root / self.cfg.embedding / 'visual'
        self.cfg.progress_path = save_root / self.cfg.embedding / 'log'

        self.cfg.ckpt_path.mkdir(exist_ok=True, parents=True)
        self.cfg.vis_path.mkdir(exist_ok=True, parents=True)
        self.cfg.progress_path.mkdir(exist_ok=True, parents=True)
        
        return self.cfg

if __name__ == "__main__":
    cfg = Config().parse(None)
    print(isinstance(cfg, SimpleNamespace))
    print(type(cfg))
