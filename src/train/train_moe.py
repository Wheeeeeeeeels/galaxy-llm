import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import AutoTokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from model import MoEModel
from model.config import ModelConfig, MoEConfig, ExpertConfig
from model import Router, DynamicRouter, LoadBalancer
from data.data_loader import get_dataloaders

// ... existing code ... 