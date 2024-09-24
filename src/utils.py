import torch
import json
import math
import os
from collections import Counter
# from src.config import CLASS_INFO_PATH, MAIN_PROJECT_DIR
from config import CLASS_INFO_PATH, MAIN_PROJECT_DIR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import List

def get_class_distribution(data_split):
    class_counts = Counter()
    for item in data_split.values():
        for region in item['regions']:
            class_name = region['region_attributes']['object_name']
            class_counts[class_name] += 1
    return class_counts

def compute_dice (gt_mask, pred_mask, class_label):
    """
    Accept tensors with shape (batch, width, height) for the masks and a tensor with shape (batch) 
    
    Args:
        gt_mask : tensor with shape (batch, width, height)
        pred_mask : tensor with shape (batch, width, height)
        class_label : tensor with shape (batch) where each element is the label of the object in the masks
    """
    gt_mask = gt_mask == class_label.unsqueeze(-1).unsqueeze(-1)
    pred_mask = pred_mask == class_label.unsqueeze(-1).unsqueeze(-1)

    intersection = torch.sum(gt_mask * pred_mask, dim=(1,2))
    total_area = torch.sum(gt_mask, dim=(1,2)) + torch.sum(pred_mask, dim=(1,2))

    return (2*intersection) / total_area

def compute_iou(gt_mask, pred_mask, class_label):
    """
    Accept tensors with shape (batch, width, height) for the masks and a tensor with shape (batch) 
    
    Args:
        gt_mask : tensor with shape (batch, width, height)
        pred_mask : tensor with shape (batch, width, height)
        class_label : tensor with shape (batch) where each element is the label of the object in the masks
    """
    gt_mask = gt_mask == class_label.unsqueeze(-1).unsqueeze(-1)
    pred_mask = pred_mask == class_label.unsqueeze(-1).unsqueeze(-1)

    intersection = torch.sum(gt_mask * pred_mask, dim=(1,2))
    union = torch.sum(gt_mask, dim=(1,2)) + torch.sum(pred_mask, dim=(1,2)) - intersection

    return intersection / (union + 1e-06)

def compute_metrics(dataset, model, processor, device="cuda"):
    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    model.to(device)
    inference_mode(model)

    running_iou = 0.0
    running_dice = 0.0

    for batch_idx, batch in enumerate(test_dataloader):
        # print(f"Iteration n. {batch_idx+1} / {len(test_dataloader)}", end="\r")
        gt_mask = dataset.get_gt_mask(batch)

        batch.pop("mask_labels", None)
        # batch.pop("text_inputs", None)
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        pred_mask = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256) for _  in range(outputs.masks_queries_logits.shape[0])]))
        running_iou += compute_iou(gt_mask, pred_mask.cpu(), batch["class_labels"][:, 1].cpu()).sum()
        running_dice += compute_dice(gt_mask, pred_mask.cpu(), batch["class_labels"][:, 1].cpu()).sum()
        
    return running_iou, running_dice


def get_id2label_mappings():
    """
    Get the id2label mapping from the class_info.json file.
    Return also label2id
    """
    with open(os.path.join(MAIN_PROJECT_DIR, CLASS_INFO_PATH), "r") as f:
        class_info = json.load(f)

    id2label = {}
    for k,v in class_info.items():
        id2label[int(k)] = v["name"]
    
    label2id = {v:k for k,v in id2label.items()}

    return id2label, label2id

class WarmupPolyLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def _get_warmup_factor_at_iter(self, method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))

    def get_lr(self) -> List[float]:
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


# Used for the validation loop where I still need the loss
def evaluation_mode(model):
    model.eval()
    for module in model.model.pixel_level_module.decoder.encoder.layers:
        module.is_training = False

# Used at test time, however it is still needed to remove mask_labels from the batches, otherwise it still tries to compute the loss (which fails)
def inference_mode(model):
    model.eval()
    model.model.is_training = False
    for module in model.model.pixel_level_module.decoder.encoder.layers:
        module.is_training = False

def training_mode(model):
    model.train()
    model.model.is_training = True
    for module in model.model.pixel_level_module.decoder.encoder.layers:
        module.is_training = True