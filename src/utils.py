import torch
import json
import os
from collections import Counter
from config import CLASS_INFO_PATH, MAIN_PROJECT_DIR


def get_class_distribution(data_split):
    class_counts = Counter()
    for item in data_split.values():
        for region in item['regions']:
            class_name = region['region_attributes']['object_name']
            class_counts[class_name] += 1
    return class_counts

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
    union = torch.sum(gt_mask + pred_mask, dim=(1,2)) - intersection

    return intersection / union

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
