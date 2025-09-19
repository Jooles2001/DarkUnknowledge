import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from accelerate import Accelerator
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os
from sklearn import linear_model, model_selection
from copy import deepcopy


from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

# Conditional import for Muon
try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Warning: Muon not installed. Install with 'pip install muon-optimizer' or from https://github.com/KellerJordan/Muon")

# Set fixed seed for reproducibility
seed = 3407  # Golden seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('runs', exist_ok=True)

# Initialize distributed process group for Muon
def init_distributed(use_muon=False):
    if use_muon and MUON_AVAILABLE:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                    init_method='tcp://localhost:29500', world_size=1, rank=0)
            print("Initialized single-process distributed group for Muon with tcp://localhost:29500")
        else:
            print("Distributed process group already initialized")
    return dist.is_initialized()

# Enhanced ResNet18CIFAR10 wrapper with proper FitNet support for timm models
class ResNet18CIFAR10(nn.Module):
    """
    Enhanced wrapper for timm resnet18_cifar10 with proper FitNet hint support.
    Adapted for timm ResNet18 structure: layer0, layer1, layer2, layer3, layer4, global_pooling, fc
    """

    def __init__(self, pretrained: bool = False, hint_layer: Optional[str] = 'layer2'):
        super(ResNet18CIFAR10, self).__init__()
        import detectors  # important for ResNet18
        import timm
        self.model = timm.create_model("resnet18_cifar10", pretrained=pretrained)

        # Initialize hint storage
        self.hint = None
        self.hint_layer = hint_layer
        self.hook_handle = None

        # Register hook for the specified layer
        if self.hint_layer is not None:
            self._register_hint_hook()

    def _register_hint_hook(self):
        """Register forward hook for hint extraction on timm ResNet18"""
        # Map hint layer names to actual timm model layers
        layer_mapping = {
            'layer0': self.model.maxpool,  # Initial conv + bn + relu
            'layer1': self.model.layer1,  # First residual block group
            'layer2': self.model.layer2,  # Second residual block group
            'layer3': self.model.layer3,  # Third residual block group
            'layer4': self.model.layer4,  # Fourth residual block group
        }

        if self.hint_layer not in layer_mapping:
            raise ValueError(f"Unsupported hint layer: {self.hint_layer}. "
                            f"Available layers: {list(layer_mapping.keys())}")

        target_layer = layer_mapping[self.hint_layer]
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Hook function to capture intermediate features"""
        # Apply global average pooling to reduce spatial dimensions for consistent hint size
        # This ensures consistent hint dimensions regardless of input size and layer depth
        if len(output.shape) == 4:  # Conv layer output [B, C, H, W]
            self.hint = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
        else:  # Already flattened
            self.hint = output.clone()

    def forward(self, x):
        """Forward pass returning both logits and hints"""
        # Reset hint before forward pass
        self.hint = None

        # Forward pass through the model (hook will capture hint)
        logits = self.model(x)

        # Ensure hint is available - create fallback if needed
        if self.hint is None:
            batch_size = x.size(0)
            # Get expected hint dimensions based on layer
            hint_dims = {
                'layer0': 64,   # After initial conv
                'layer1': 64,   # First block group
                'layer2': 128,  # Second block group
                'layer3': 256,  # Third block group
                'layer4': 512,  # Fourth block group
            }

            hint_dim = hint_dims.get(self.hint_layer, 512)
            self.hint = torch.zeros(batch_size, hint_dim, device=x.device)
            # print(f"Warning: Using fallback zero hint for {self.hint_layer}")

        return logits, self.hint.clone()  # Clone to prevent reference issues

    def cleanup_hooks(self):
        """Clean up hooks to prevent memory leaks"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class FitNetTrainer:
    """
    Proper FitNet implementation with two-stage training as described in the original paper.
    """
    def __init__(self, teacher_model, student_model, hint_layer='layer2'):
        self.teacher = teacher_model
        self.student = student_model
        self.hint_layer = hint_layer

        # Ensure both models use the same hint layer
        if hasattr(self.teacher, 'hint_layer'):
            self.teacher.hint_layer = hint_layer
        if hasattr(self.student, 'hint_layer'):
            self.student.hint_layer = hint_layer

    def stage1_hint_training(self, dataloader, epochs=20, lr=0.01, device='cpu',
                            accelerator=None, writer=None, tag='stage1', use_muon=False):
        """
        Stage 1: Train student to match teacher's intermediate representations (hints)
        """
        print(f"Starting FitNet Stage 1: Hint Training for {epochs} epochs")

        # Get unwrapped model for optimizer
        student_unwrapped = accelerator.unwrap_model(self.student) if accelerator else self.student
        optimizer = get_optimizer(student_unwrapped, lr, use_muon)

        if accelerator:
            self.student, optimizer, dataloader = accelerator.prepare(
                self.student, optimizer, dataloader
            )

        scheduler = None if use_muon else ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.001)
        criterion_hint = nn.MSELoss()

        self.teacher.eval()
        self.student.train()

        pbar = tqdm(range(epochs), desc=f"FitNet Stage 1 - {tag}")
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0
            data_pbar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, (data, target) in enumerate(data_pbar):
                if device != 'cpu' and not accelerator:
                    data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    _, teacher_hint = self.teacher(data)
                    teacher_hint = F.normalize(teacher_hint, p=2, dim=1)

                _, student_hint = self.student(data)
                student_hint = F.normalize(student_hint, p=2, dim=1)
                loss_hint = criterion_hint(student_hint, teacher_hint.detach())

                if not torch.isfinite(loss_hint):
                    print(f"Warning: NaN loss in Stage 1, batch {batch_idx}")
                    continue

                if accelerator:
                    accelerator.backward(loss_hint)
                else:
                    loss_hint.backward()

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss_hint.item()
                num_batches += 1
                data_pbar.set_postfix({'hint_loss': loss_hint.item()})

            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            if scheduler is not None:
                scheduler.step(avg_loss)
            pbar.set_postfix({'avg_hint_loss': avg_loss})

            if writer:
                writer.add_scalar(f'{tag}/hint_loss', avg_loss, epoch)
                if not use_muon:
                    writer.add_scalar(f'{tag}/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print("Stage 1 (Hint Training) completed")
        return accelerator.unwrap_model(self.student) if accelerator else self.student

    def stage2_knowledge_distillation(self, dataloader, epochs=100, lr=0.01, T=4.0,
                                    alpha=0.9, hint_weight=0.1, device='cpu',
                                    accelerator=None, writer=None, tag='stage2',
                                    val_loader=None, milestones=None, use_muon=False):
        """
        Stage 2: Combined hint matching and knowledge distillation
        """
        print(f"Starting FitNet Stage 2: Knowledge Distillation for {epochs} epochs")

        student_unwrapped = accelerator.unwrap_model(self.student) if accelerator else self.student
        optimizer = get_optimizer(student_unwrapped, lr, use_muon)

        if accelerator:
            self.student, optimizer, dataloader = accelerator.prepare(self.student, optimizer, dataloader)

        if use_muon:
            scheduler = None
        elif milestones is None:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, threshold=0.001, verbose=True)
        else:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        criterion_hard = nn.CrossEntropyLoss()
        criterion_hint = nn.MSELoss()
        self.teacher.eval()
        self.student.train()
        pbar = tqdm(range(epochs), desc=f"FitNet Stage 2 - {tag}")
        for epoch in pbar:
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
            epoch_soft_loss, epoch_hard_loss, epoch_hint_loss = 0.0, 0.0, 0.0
            data_pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, (data, target) in enumerate(data_pbar):
                if device != 'cpu' and not accelerator:
                    data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits, teacher_hint = self.teacher(data)
                    teacher_logits = torch.clamp(teacher_logits, min=-100, max=100)
                    teacher_hint = F.normalize(teacher_hint, p=2, dim=1)

                student_logits, student_hint = self.student(data)
                student_logits = torch.clamp(student_logits, min=-100, max=100)
                student_hint = F.normalize(student_hint, p=2, dim=1)

                teacher_soft = F.softmax(teacher_logits / T, dim=1)
                student_soft = F.log_softmax(student_logits / T, dim=1)
                loss_soft = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
                loss_hard = criterion_hard(student_logits, target)
                loss_hint = criterion_hint(student_hint, teacher_hint.detach())
                total_loss = alpha * loss_soft + (1 - alpha) * loss_hard + hint_weight * loss_hint

                if not torch.isfinite(total_loss):
                    print(f"Warning: NaN loss in Stage 2, batch {batch_idx}")
                    continue

                if accelerator:
                    accelerator.backward(total_loss)
                else:
                    total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()

                pred = student_logits.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_total += target.size(0)
                epoch_loss += total_loss.item()
                epoch_soft_loss += loss_soft.item()
                epoch_hard_loss += loss_hard.item()
                epoch_hint_loss += loss_hint.item()

                data_pbar.set_postfix({'total_loss': total_loss.item(), 'batch_acc': pred.eq(target).sum().item() / target.size(0)})

            avg_loss = epoch_loss / len(dataloader)
            epoch_acc = epoch_correct / epoch_total
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_acc)
                else:
                    scheduler.step()
            pbar.set_postfix({'total_loss': avg_loss, 'train_acc': epoch_acc})

            if writer:
                writer.add_scalar(f'{tag}/total_loss', avg_loss, epoch)
                writer.add_scalar(f'{tag}/train_acc', epoch_acc, epoch)

            if val_loader and (epoch % 10 == 0 or epoch == epochs - 1):
                val_acc_model = accelerator.unwrap_model(self.student) if accelerator else self.student
                val_acc, val_loss = compute_val_acc(val_acc_model, val_loader, device)
                if writer:
                    writer.add_scalar(f'{tag}/val_acc', val_acc, epoch)
                    writer.add_scalar(f'{tag}/val_loss', val_loss, epoch)
                print(f"{tag} Epoch {epoch+1}/{epochs} - Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        print("Stage 2 (Knowledge Distillation) completed")
        return accelerator.unwrap_model(self.student) if accelerator else self.student

# Load model checkpoint
def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {path}")
        return True
    return False

# Save model checkpoint
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved {path}")

# Get optimizer (Muon or AdamW)
def get_optimizer(model, lr=0.01, use_muon=False):
    if use_muon and MUON_AVAILABLE:
        hidden_weights, hidden_gains_biases, nonhidden_params = [], [], []
        for name, param in model.named_parameters():
            if 'conv1' in name or 'fc' in name or 'classifier' in name:
                nonhidden_params.append(param)
            elif param.ndim >= 2:
                hidden_weights.append(param)
            else:
                hidden_gains_biases.append(param)
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.02),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, lr=lr, betas=(0.9, 0.95), weight_decay=0.02),
        ]
        print(f"Using MuonWithAuxAdam optimizer: {len(hidden_weights)} hidden, {len(hidden_gains_biases) + len(nonhidden_params)} non-hidden params")
        return MuonWithAuxAdam(param_groups)
    else:
        print("Using AdamW optimizer")
        return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

# Accuracy function for validation during training
def compute_val_acc(model, val_loader, device):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = torch.clamp(output, min=-100, max=100)
            loss += F.cross_entropy(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return (correct / total), (loss.item() / total)

# Training function
def train_model(accelerator, model, dataloader, epochs=50, lr=0.1, milestones=None, device='cpu', writer=None, tag='', val_loader=None, use_muon=False):
    optimizer = get_optimizer(accelerator.unwrap_model(model), lr=lr, use_muon=use_muon)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if use_muon:
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, threshold=0.001)
        
    elif milestones is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=0.001)
    else:
        scheduler = MultiStepLR(optimizer, milestones=milestones or [82, 122], gamma=0.1)

    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    pbar = tqdm(range(epochs), desc=f"Training {tag}")
    for epoch in pbar:
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        data_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in data_pbar:
            optimizer.zero_grad()
            output, _ = model(data)
            output = torch.clamp(output, min=-100, max=100)
            # loss = F.cross_entropy(output, target)
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                print(f"Warning: NaN loss detected in {tag}, skipping batch")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pred = output.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total += target.size(0)
            epoch_loss += loss.item()
            data_pbar.set_postfix({'batch_loss': loss.item(), 'batch_acc': pred.eq(target).sum().item() / target.size(0)})

        val_acc, val_loss = compute_val_acc(accelerator.unwrap_model(model), val_loader, device)
        writer.add_scalar(f'{tag}/val_acc', val_acc, epoch)
        writer.add_scalar(f'{tag}/val_loss', val_loss, epoch)
        print(f"{tag} Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()
            if writer:
                writer.add_scalar(f'{tag}/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        avg_loss = epoch_loss / len(dataloader)
        epoch_acc = epoch_correct / epoch_total
        pbar.set_postfix({'epoch_loss': avg_loss, 'epoch_acc': epoch_acc})
        if writer:
            writer.add_scalar(f'{tag}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'{tag}/train_acc', epoch_acc, epoch)

        # if val_loader and (epoch % 5 == 0 or epoch == epochs - 1):
            
    return accelerator.unwrap_model(model)

# Distillation function
def distill_model(accelerator, teacher_model, student, dataloader, epochs=164, lr=0.01, T=4.0, alpha=0.9, distill_method='vanilla', hint_weight=0.1, milestones=None, device='cpu', writer=None, tag='', val_loader=None, use_muon=False, stage1_epochs=20):
    if T < 1.0:
        print(f"Warning: Temperature T={T} is low. Consider T >= 1.0.")

    if distill_method == 'fitnets':
        fitnet_trainer = FitNetTrainer(teacher_model, student, hint_layer='layer2')
        fitnet_trainer.stage1_hint_training(dataloader, stage1_epochs, lr * 2, device, accelerator, writer, f'{tag}_fitnet_stage1', use_muon)
        return fitnet_trainer.stage2_knowledge_distillation(dataloader, epochs - stage1_epochs, lr, T, alpha, hint_weight, device, accelerator, writer, f'{tag}_fitnet_stage2', val_loader, milestones, use_muon)

    # Vanilla KD
    optimizer = get_optimizer(accelerator.unwrap_model(student), lr=lr, use_muon=use_muon)
    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)

    if use_muon:
        scheduler = None
    elif milestones is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=0.001)
    else:
        scheduler = MultiStepLR(optimizer, milestones=milestones or [82, 122], gamma=0.1)

    criterion_hard = nn.CrossEntropyLoss()
    student.train()
    teacher_model.eval()
    pbar = tqdm(range(epochs), desc=f"Distilling {tag}")
    for epoch in pbar:
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        data_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in data_pbar:
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits, _ = teacher_model(data)
                teacher_logits = torch.clamp(teacher_logits, min=-100, max=100)

            student_logits, _ = student(data)
            student_logits = torch.clamp(student_logits, min=-100, max=100)

            teacher_out = F.softmax(teacher_logits / T, dim=1)
            student_out = F.log_softmax(student_logits / T, dim=1)
            loss_soft = F.kl_div(student_out, teacher_out, reduction='batchmean') * (T ** 2)
            loss_hard = criterion_hard(student_logits, target)
            loss = alpha * loss_soft + (1 - alpha) * loss_hard

            if not torch.isfinite(loss):
                print(f"Warning: NaN loss detected in {tag}, skipping batch")
                continue

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            pred = student_logits.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total += target.size(0)
            epoch_loss += loss.item()
            data_pbar.set_postfix({'batch_loss': loss.item(), 'batch_acc': pred.eq(target).sum().item() / target.size(0)})

        if scheduler is not None:
            scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        epoch_acc = epoch_correct / epoch_total
        pbar.set_postfix({'epoch_loss': avg_loss, 'epoch_acc': epoch_acc})
        if writer:
            writer.add_scalar(f'{tag}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'{tag}/train_acc', epoch_acc, epoch)

        if val_loader and (epoch % 20 == 0 or epoch == epochs - 1):
            val_acc, val_loss = compute_val_acc(accelerator.unwrap_model(student), val_loader, device)
            writer.add_scalar(f'{tag}/val_acc', val_acc, epoch)
            writer.add_scalar(f'{tag}/val_loss', val_loss, epoch)
            print(f"{tag} Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")
    return accelerator.unwrap_model(student)

# Unlearning function
def unlearn_model(accelerator, model, retain_loader, forget_loader, epochs=10, lr=0.001, device='cpu', writer=None, tag=''):
    model, optimizer, retain_loader, forget_loader = accelerator.prepare(
        model, optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4), retain_loader, forget_loader
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    pbar = tqdm(range(epochs), desc=f"Unlearning {tag}")
    for epoch in pbar:
        epoch_forget_loss, epoch_retain_loss = 0.0, 0.0
        data_pbar_forget = tqdm(forget_loader, desc=f"Epoch {epoch+1}/{epochs} (Forget)", leave=False)
        for data, target in data_pbar_forget:
            optimizer.zero_grad()
            output, _ = model(data)
            output = torch.clamp(output, min=-100, max=100)
            loss = -criterion(output, target)
            if not torch.isfinite(loss):
                print(f"Warning: NaN loss in {tag} (forget), skipping batch")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_forget_loss += loss.item()
            data_pbar_forget.set_postfix({'forget_loss': loss.item()})

        data_pbar_retain = tqdm(retain_loader, desc=f"Epoch {epoch+1}/{epochs} (Retain)", leave=False)
        for data, target in data_pbar_retain:
            optimizer.zero_grad()
            output, _ = model(data)
            output = torch.clamp(output, min=-100, max=100)
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                print(f"Warning: NaN loss in {tag} (retain), skipping batch")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_retain_loss += loss.item()
            data_pbar_retain.set_postfix({'retain_loss': loss.item()})

        avg_forget_loss = epoch_forget_loss / len(forget_loader)
        avg_retain_loss = epoch_retain_loss / len(retain_loader)
        pbar.set_postfix({'forget_loss': avg_forget_loss, 'retain_loss': avg_retain_loss})
        if writer:
            writer.add_scalar(f'{tag}/forget_loss', avg_forget_loss, epoch)
            writer.add_scalar(f'{tag}/retain_loss', avg_retain_loss, epoch)
    return accelerator.unwrap_model(model)

# --- Evaluation Functions (Modified to always handle lists) ---
def ensemble_forward(models, x):
    if not isinstance(models, list):
        raise TypeError("Input to ensemble_forward must be a list of models.")
    logits_list, hints_list = [], []
    for m in models:
        logits, hint = m(x)
        logits_list.append(logits)
        hints_list.append(hint)
    return torch.stack(logits_list).mean(0), torch.stack(hints_list).mean(0)

def accuracy(model_list, dataloader, device='cpu'):
    for m in model_list: m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = ensemble_forward(model_list, data)[0]
            output = torch.clamp(output, min=-100, max=100)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


def unlearning_score(model_list, retain_loader, forget_loader, test_loader=None, device='cpu'):
    ra = accuracy(model_list, retain_loader, device)
    fa = accuracy(model_list, forget_loader, device)
    if test_loader:
        ta = accuracy(model_list, test_loader, device)
    return ra, fa, ra - fa

def individual_model_distance(m1, m2):
    """Computes normalized L2 distance between two SINGLE models."""
    params1 = torch.cat([p.data.view(-1) for p in m1.parameters()]).cpu().numpy()
    params2 = torch.cat([p.data.view(-1) for p in m2.parameters()]).cpu().numpy()
    l2_dist = np.linalg.norm(params1 - params2)
    norm = np.sqrt(len(params1))
    return l2_dist / norm

def get_single_model_activations(model, loader, device, subsample=1000):
    """Retrieves the intermediate activations for a SINGLE model."""
    model.eval()
    feats = []
    idx = 0
    with torch.no_grad():
        for data, _ in loader:
            if idx >= subsample:
                break
            data = data.to(device)
            _, hint = model(data)  # Get hint (intermediate activation)
            feats.append(hint.detach().cpu().numpy())
            idx += data.size(0)
    return np.vstack(feats)

def compute_cka(acts1, acts2):
    acts1 = acts1 - np.mean(acts1, axis=0, keepdims=True)
    acts2 = acts2 - np.mean(acts2, axis=0, keepdims=True)
    K1 = acts1 @ acts1.T
    K2 = acts2 @ acts2.T
    n = K1.shape[0]
    if n == 0: return 0.0
    H = np.eye(n) - (1 / n) * np.ones((n, n))
    K1c, K2c = H @ K1 @ H, H @ K2 @ H
    hsic12 = (1 / n**2) * np.trace(K1c @ K2c)
    hsic11 = (1 / n**2) * np.trace(K1c @ K1c)
    hsic22 = (1 / n**2) * np.trace(K2c @ K2c)
    return hsic12 / np.sqrt(hsic11 * hsic22 + 1e-8)

# --- Visualization and MIA functions (unchanged from previous version) ---
def plot_distances(model_names, dist_matrix, title='Distance Matrix', cmap='viridis'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix, xticklabels=model_names, yticklabels=model_names, annot=True, fmt=".3f", cmap=cmap)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


class ModeConnectivity:
    """
    Computes and evaluates the linear path between two models.
    """
    def __init__(self, model_a, model_b, loader, device, hint_layer=None):
        self.model_a = model_a.to(device)
        self.model_b = model_b.to(device)
        self.loader = loader
        self.device = device
        self.architecture = type(model_a)
        self.hint_layer = hint_layer

    def _get_weights(self, model):
        """Flattens all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _set_weights(self, model, weights_vec):
        """Loads a flat weight vector into a model."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            # Ensure slicing doesn't go out of bounds
            if offset + numel > len(weights_vec):
                raise ValueError("The provided weights vector is smaller than the model's total number of parameters.")
            p.data.copy_(weights_vec[offset:offset + numel].view_as(p.data))
            offset += numel

    def evaluate_point(self, model):
        """Evaluates a model for one epoch to get loss and accuracy."""
        model.eval()
        total_loss, correct, total = 0, 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return total_loss / total, correct / total

    def compute_linear_path(self, steps=11):
        """Computes loss and accuracy along the linear path between model_a and model_b."""
        theta_a = self._get_weights(self.model_a)
        theta_b = self._get_weights(self.model_b)

        results = []
        pbar = tqdm(range(steps), desc="Computing Mode Connectivity")
        for i in pbar:
            t = i / (steps - 1)
            interpolated_theta = (1 - t) * theta_a + t * theta_b

            # Create a new model instance for evaluation
            temp_model = self.architecture(pretrained=False, hint_layer=self.hint_layer).to(self.device)
            self._set_weights(temp_model, interpolated_theta)

            loss, acc = self.evaluate_point(temp_model)
            results.append({'t': t, 'loss': loss, 'acc': acc})
            pbar.set_postfix({'t': f"{t:.2f}", 'loss': f"{loss:.4f}", 'acc': f"{acc:.4f}"})

        return results

def plot_mode_connectivity(results, title=""):
    """Plots the loss and accuracy curves from the connectivity results."""
    ts = [r['t'] for r in results]
    losses = [r['loss'] for r in results]
    accs = [r['acc'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Interpolation (t) [Model A to Model B]')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(ts, losses, color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(ts, accs, color=color, marker='x', linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0) # Accuracy shouldn't be negative

    plt.title(f'Mode Connectivity: {title}')
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KD and Unlearning on CIFAR10 with Ensembles')
    parser.add_argument('--student_ensemble_size', type=int, default=3, help='Number of models in student ensembles.')
    parser.add_argument('--use_muon', action='store_true', help='Use Muon optimizer instead of AdamW')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs for teachers (use 164+ for 95%% acc)')
    parser.add_argument('--alpha', type=float, default=0.9, help='Distillation alpha (soft loss weight)')
    parser.add_argument('--T', type=float, default=4.0, help='Distillation temperature')
    parser.add_argument('--distill_method', type=str, default='vanilla', choices=['vanilla', 'fitnets'], help='Distillation method: vanilla KD or FitNets with hints')
    parser.add_argument('--distill_epochs', type=int, default=20, help='Distillation epochs for students')
    parser.add_argument('--hint_weight', type=float, default=0.1, help='Weight for hint loss in FitNets')
    parser.add_argument('--hint_layer', type=str, default='layer2', choices=['layer0', 'layer1', 'layer2', 'layer3', 'layer4'], help='Layer for hint extraction in FitNets')
    parser.add_argument('--stage1_epochs', type=int, default=10, help='FitNet Stage 1 (hint training) epochs')
    parser.add_argument('--unlearn_epochs', type=int, default=10, help='Unlearning epochs')
    parser.add_argument('--run_mode_connectivity', action='store_true', help='Run (potentially slow) mode connectivity analysis.')
    args = parser.parse_args()

    init_distributed(args.use_muon)
    accelerator = Accelerator()
    device = accelerator.device
    milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
    writer = SummaryWriter(log_dir=f'runs/unlearning_exp_ens{args.student_ensemble_size}_{args.distill_method}')

    try:
        # Data preparation
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        n_forget = int(0.1 * len(train_dataset))
        forget_idx = np.random.choice(len(train_dataset), n_forget, replace=False)
        retain_idx = np.setdiff1d(np.arange(len(train_dataset)), forget_idx)
        retain_dataset = Subset(train_dataset, retain_idx)
        retain_loader = DataLoader(retain_dataset, batch_size=256, shuffle=True)
        forget_loader = DataLoader(Subset(datasets.CIFAR10('./data', train=True, transform=transform_test), forget_idx), batch_size=256, shuffle=False)
        full_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        retain_eval_loader = DataLoader(Subset(datasets.CIFAR10('./data', train=True, transform=transform_test), retain_idx), batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        small_test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        val_loader = test_loader

        # --- Teacher Models (Single Instances) ---
        print("\n--- Preparing Teacher Models ---")
        teacher_full = ResNet18CIFAR10(pretrained=True, hint_layer=args.hint_layer).to(device)
        print("Loaded ONE pretrained timm resnet18_cifar10 as TeacherFull.")
        # Evaluate TeacherFull
        teacher_full_acc, _ = compute_val_acc(teacher_full, test_loader, device)
        print(f"TeacherFull Test Accuracy: {teacher_full_acc:.4f}")
        
        teacher_retain_path = 'checkpoints/teacher_retain.pth'
        teacher_retain = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
        if not load_model(teacher_retain, teacher_retain_path, device):
            print("Training ONE TeacherRetain model (golden standard)...")
            teacher_retain = train_model(accelerator, teacher_retain, retain_loader, args.epochs, 0.01, milestones, device, writer, 'teacher_retain', val_loader, args.use_muon)
            save_model(teacher_retain, teacher_retain_path)
        teacher_retain.to(device)
        teacher_retain_acc, _ = compute_val_acc(teacher_retain, test_loader, device)
        print(f"TeacherRetain Test Accuracy: {teacher_retain_acc:.4f}")

        # --- Build All Ensembles ---
        print(f"\n--- Building All Ensembles (size={args.student_ensemble_size}) ---")
        
        ### NEW ENSEMBLE ###: UnlearnedTeacher Ensemble
        unlearned_teacher_ensemble = []
        pbar = tqdm(range(args.student_ensemble_size), desc="Creating UnlearnedTeacher Ensemble")
        for i in pbar:
            path = f'checkpoints/unlearned_teacher_{i}.pth'
            model = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model, path, device):
                model.load_state_dict(deepcopy(teacher_full.state_dict())) # Start from the same TeacherFull
                model = unlearn_model(accelerator, model, retain_loader, forget_loader, args.unlearn_epochs, 0.001, device, writer, f'unlearned_teacher_{i}')
                save_model(model, path)
            unlearned_teacher_ensemble.append(model.to(device))
            # print(f"UnlearnedTeacher {i} Test Acc: {compute_val_acc(model, test_loader, device):.4f}")
        print(f"UnlearnedTeacher Ensemble Average Test Acc: {accuracy(unlearned_teacher_ensemble, test_loader, device):.4f}")
        
        ### NEW ENSEMBLE ###: Student from UnlearnedTeacher Ensemble
        student_unlearned_teacher_ensemble = []
        pbar = tqdm(range(args.student_ensemble_size), desc="Distilling StudentUnlearnedTeacher Ens")
        for i in pbar:
            path = f'checkpoints/student_unlearned_teacher_{args.distill_method}_{i}.pth'
            model = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model, path, device):
                teacher = unlearned_teacher_ensemble[i] # One-to-one distillation
                model = distill_model(accelerator, teacher, model, retain_loader, args.distill_epochs, 0.01, args.T, args.alpha, args.distill_method, args.hint_weight, None, device, writer, f'student_unlearned_teacher_{i}', val_loader, args.use_muon, args.stage1_epochs)
                save_model(model, path)
            student_unlearned_teacher_ensemble.append(model.to(device))
            # print(f"StudentUnlearnedTeacher {i} Test Acc: {compute_val_acc(model, test_loader, device):.4f}")
        print(f"StudentUnlearnedTeacher Ensemble Average Test Acc: {accuracy(student_unlearned_teacher_ensemble, test_loader, device):.4f}")

        ### NEW ENSEMBLE ###: StudentFullRetain Ensemble (Fine-tuning baseline)
        student_full_retain_ensemble = []
        pbar = tqdm(range(args.student_ensemble_size), desc="Distilling StudentFullRetain Ensemble")
        for i in pbar:
            path = f'checkpoints/student_full_retain_{args.distill_method}_{i}.pth'
            model = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model, path, device):
                # Distill from TeacherFull but only on retain data
                model = distill_model(accelerator, teacher_full, model, retain_loader, args.distill_epochs, 0.01, args.T, args.alpha, args.distill_method, args.hint_weight, None, device, writer, f'student_full_retain_{i}', val_loader, args.use_muon, args.stage1_epochs)
                save_model(model, path)
            student_full_retain_ensemble.append(model.to(device))
            # print(f"StudentFullRetain {i} Test Acc: {compute_val_acc(model, test_loader, device):.4f}")
        print(f"StudentFullRetain Ensemble Average Test Acc: {accuracy(student_full_retain_ensemble, test_loader, device):.4f}")

        # Previously existing ensembles
        student_full_ensemble, unlearned_student_ensemble, student_dist_retain_ensemble = [], [], []
        for i in range(args.student_ensemble_size):
            # StudentFull
            path = f'checkpoints/student_full_{args.distill_method}_{args.hint_layer}_{i}.pth'
            model = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model, path, device):
                model = distill_model(accelerator, teacher_full, model, full_train_loader, args.distill_epochs, 0.01, args.T, args.alpha, args.distill_method, args.hint_weight, None, device, writer, f'student_full_{i}', val_loader, args.use_muon, args.stage1_epochs)
                save_model(model, path)
            student_full_ensemble.append(model.to(device))
            # print(f"StudentFull {i} Test Acc: {compute_val_acc(model, test_loader, device):.4f}")

            # UnlearnedStudent
            path = f'checkpoints/unlearned_student_{args.distill_method}_{args.hint_layer}_{i}.pth'
            model_unlearn = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model_unlearn, path, device):
                model_unlearn.load_state_dict(deepcopy(student_full_ensemble[i].state_dict()))
                model_unlearn = unlearn_model(accelerator, model_unlearn, retain_loader, forget_loader, args.unlearn_epochs, 0.001, device, writer, f'unlearned_student_{i}')
                save_model(model_unlearn, path)
            unlearned_student_ensemble.append(model_unlearn.to(device))
            # print(f"UnlearnedStudent {i} Test Acc: {compute_val_acc(model_unlearn, test_loader, device):.4f}")
            
            # StudentDistRetain
            path = f'checkpoints/student_dist_retain_{args.distill_method}_{args.hint_layer}_{i}.pth'
            model_dist_retain = ResNet18CIFAR10(pretrained=False, hint_layer=args.hint_layer)
            if not load_model(model_dist_retain, path, device):
                model_dist_retain = distill_model(accelerator, teacher_retain, model_dist_retain, retain_loader, args.distill_epochs, 0.01, args.T, args.alpha, args.distill_method, args.hint_weight, None, device, writer, f'student_dist_retain_{i}', val_loader, args.use_muon, args.stage1_epochs)
                save_model(model_dist_retain, path)
            student_dist_retain_ensemble.append(model_dist_retain.to(device))
            # print(f"StudentDistRetain {i} Test Acc: {compute_val_acc(model_dist_retain, test_loader, device):.4f}")
        print(f"StudentFull Ensemble Average Test Acc: {accuracy(student_full_ensemble, test_loader, device):.4f}")
        print(f"UnlearnedStudent Ensemble Average Test Acc: {accuracy(unlearned_student_ensemble, test_loader, device):.4f}")
        print(f"StudentDistRetain Ensemble Average Test Acc: {accuracy(student_dist_retain_ensemble, test_loader, device):.4f}")

        # --- Model Dictionary for Evaluation ---
        models = {
            'TeacherFull': [teacher_full],
            'TeacherRetain': [teacher_retain],
            'UnlearnedTeacher': unlearned_teacher_ensemble,
            'StudentFull': student_full_ensemble,
            'StudentFullRetain': student_full_retain_ensemble,
            'UnlearnedStudent': unlearned_student_ensemble,
            'StudentUnlearnedTeacher': student_unlearned_teacher_ensemble,
            'StudentDistRetain': student_dist_retain_ensemble,
            'RandomModel': [ResNet18CIFAR10(pretrained=False, hint_layer=None).to(device)],
        }
        model_names = list(models.keys())

        # --- Evaluation ---
        print("\n" + "="*60 + "\nMODEL SIMILARITY ANALYSIS (INDIVIDUAL MODELS)\n" + "="*60)
        # 1. Unroll the ensembles into a flat list of every individual model
        expanded_models = []
        expanded_model_names = []
        for name, m_list in models.items():
            if len(m_list) > 1:
                for i, m in enumerate(m_list):
                    expanded_models.append(m)
                    expanded_model_names.append(f"{name}_{i}")
            else:
                # Keep single models as they are (e.g., teachers)
                expanded_models.append(m_list[0])
                expanded_model_names.append(name)

        num_individual_models = len(expanded_models)
        dist_matrix = np.zeros((num_individual_models, num_individual_models))
        cka_matrix = np.zeros((num_individual_models, num_individual_models))

        # 2. Pre-compute activations for all models to be efficient
        print("Pre-computing all model activations for CKA analysis...")
        all_activations = [get_single_model_activations(m, small_test_loader, device) for m in tqdm(expanded_models, desc="Getting activations")]

        # 3. Iterate through every unique pair of individual models
        pbar_dist = tqdm(list(combinations(range(num_individual_models), 2)), desc="Computing Pairwise Similarities")
        for i, j in pbar_dist:
            model1 = expanded_models[i]
            model2 = expanded_models[j]
            name1 = expanded_model_names[i]
            name2 = expanded_model_names[j]

            # L2 distance between model parameters
            l2 = individual_model_distance(model1, model2)
            dist_matrix[i, j] = dist_matrix[j, i] = l2

            # CKA similarity between intermediate representations
            acts1 = all_activations[i]
            acts2 = all_activations[j]
            cka_sim = compute_cka(acts1, acts2)
            cka_matrix[i, j] = cka_matrix[j, i] = cka_sim
            
            # Optional: uncomment to print individual results
            # print(f"  {name1:25s} vs {name2:25s}: L2={l2:.4f}, CKA={cka_sim:.4f}")

        # 4. Fill diagonals and plot the detailed heatmaps
        np.fill_diagonal(dist_matrix, 0)
        np.fill_diagonal(cka_matrix, 1)
        print("Plotting similarity matrices for all individual models...")
        plot_distances(expanded_model_names, dist_matrix, 'Normalized L2 Distance Matrix (Individual Models)', 'viridis')
        plot_distances(expanded_model_names, cka_matrix, 'CKA Similarity Matrix (Individual Models)', 'inferno')

        # The Mode Connectivity section already compares individual models, so it remains unchanged.
        if args.run_mode_connectivity:
            print("\n" + "="*60 + "\nMODE CONNECTIVITY ANALYSIS\n" + "="*60)

            # Pair 1: Unlearning method vs. Distilled "Retraining" (ideal)
            print("\nConnectivity: UnlearnedStudent vs StudentDistRetain (Efficacy Check)")
            mc1 = ModeConnectivity(models['UnlearnedStudent'][0], models['StudentDistRetain'][0], retain_eval_loader, device, args.hint_layer)
            res1 = mc1.compute_linear_path(steps=11)
            plot_mode_connectivity(res1, title="UnlearnedStudent[0] vs. StudentDistRetain[0]")

            # Pair 2: Fine-tuning unlearning vs. "Retraining" (baseline)
            print("\nConnectivity: StudentFullRetain vs StudentDistRetain (Baseline Check)")
            mc2 = ModeConnectivity(models['StudentFullRetain'][0], models['StudentDistRetain'][0], retain_eval_loader, device, args.hint_layer)
            res2 = mc2.compute_linear_path(steps=11)
            plot_mode_connectivity(res2, title="StudentFullRetain[0] vs. StudentDistRetain[0]")

            # Pair 3: Diversity check within an ensemble
            if args.student_ensemble_size > 1:
                print("\nConnectivity: Within StudentFull Ensemble (Diversity Check)")
                mc3 = ModeConnectivity(models['StudentFull'][0], models['StudentFull'][1], test_loader, device, args.hint_layer)
                res3 = mc3.compute_linear_path(steps=11)
                plot_mode_connectivity(res3, title="StudentFull[0] vs. StudentFull[1]")

    except Exception as e:
        print(f"\nError during experiment: {e}")
        raise e
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        writer.close()
