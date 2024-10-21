import torch
import json
import math
import os
import skimage
import numpy as np
from collections import Counter
# from src.config import CLASS_INFO_PATH, MAIN_PROJECT_DIR
from config import CLASS_INFO_PATH, MAIN_PROJECT_DIR, GALAXY_MEAN, GALAXY_STD
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
        class_label : tensor with shape (batch) where each element is the label of the object in the mask
    """
    gt_mask = gt_mask == class_label.unsqueeze(-1).unsqueeze(-1)
    pred_mask = pred_mask == class_label.unsqueeze(-1).unsqueeze(-1)

    intersection = torch.sum(gt_mask * pred_mask, dim=(1,2))
    union = torch.sum(gt_mask, dim=(1,2)) + torch.sum(pred_mask, dim=(1,2)) - intersection

    return intersection / (union + 1e-06)

def sliding_window(image, window_size, step_size):
    """Generates sliding window patches from the image."""
    img_width, img_height = image.shape[:2]
    window_height, window_width = window_size
    step_height, step_width = step_size

    for y in range(0, img_height - window_height + 1, step_height):
        for x in range(0, img_width - window_width + 1, step_width):
            yield x, y, image[y:y + window_height, x:x + window_width]

def extract_bounding_box(segmentation_map, offset=(0,0)):
    """
        Return x, y, widht, height w.r.t. to the original image coordinates
    """
    if segmentation_map.shape[0] == 1:
        segmentation_map = segmentation_map.squeeze()
    
    object_coords = torch.nonzero(segmentation_map > 0, as_tuple=False)    
    if object_coords.size(0) == 0:
        return None
    
    y_min, x_min = torch.min(object_coords, dim=0).values
    y_max, x_max = torch.max(object_coords, dim=0).values
    
    x_max += offset[0]
    x_min += offset[0]
    y_max += offset[1]
    y_min += offset[1]

    
    width = (x_max.item() - x_min.item())+1
    height = (y_max.item() - y_min.item())+1

    return (x_min.item(), y_min.item(), width, height)

def nms(predictions, iou_thresh, task):
    # NMS
    if task == "instance":
        ordered_pred = sorted(predictions, key=lambda x:x["score"], reverse=True)
    elif task == "semantic":
        ordered_pred = predictions
    keep = []

    print(f"{len(ordered_pred)} objects extracted")
    print(f"Performing NMS...")
    while len(ordered_pred) > 0:
        if len(ordered_pred) == 0:
            break

        # pop first element
        predA = ordered_pred.pop(0)
        keep.append(predA)
        
        ordered_pred_copy = ordered_pred[:]

        # mark element to delete
        to_delete = []    
        for indexB in range(len(ordered_pred_copy)):
            boxA = extract_bounding_box(predA["mask"], predA["pos"])
            boxB = extract_bounding_box(ordered_pred_copy[indexB]["mask"], ordered_pred_copy[indexB]["pos"])
            iou = bb_intersection_over_union(boxA, boxB)

            if iou >= iou_thresh:
                # print(f"popping: {indexB}")
                to_delete.append(indexB)

        # delete elements
        ordered_pred = [ordered_pred[i] for i in range(len(ordered_pred)) if i not in to_delete]

    print(f"{len(keep)} objects after NMS")
    return keep

def run_sliding_windows_over_image(img_path, model, processor, step_size, conf_thresh, pixel_thresh, task="instance"):
    image = skimage.io.imread(img_path)

    # Image dimensions
    img_width, img_height = image.shape[:2]

    # Window size
    window_height, window_width = 256, 256

    # Step size (how far the window slides)
    step_height, step_width = step_size, step_size

    predictions = []

    n_chunks = ((img_width-window_width)//step_width) * ((img_height-window_height)//step_height)
    print(f"Sliding window...")
    print(f"{n_chunks} chunks being processed")
    
    model.to("cuda")
    inference_mode(model)
    
    # Create sliding windows over the image
    for (x, y, window) in sliding_window(image, (window_width, window_height), (step_width, step_height)):
        # print(f"Window position: x={x}, y={y}, Window shape: {window.shape}")
        mean = window.mean(axis=(0,1))
        std = window.std(axis=(0,1))
        
        encoded_inputs = processor(
            images=window,
            task_inputs=[task],
            return_tensors="pt",
            image_mean=mean, 
            image_std=std
        )    

        input = {k:v.to("cuda") for k,v in encoded_inputs.items()}

        with torch.no_grad():
            outputs = model(**input)
        
        if task == "instance":
            pred_inst = processor.post_process_instance_segmentation(outputs, overlap_mask_area_threshold=0.5, target_sizes=[(256,256) for _ in range(outputs.masks_queries_logits.shape[0])])
            converted = segments_to_label(pred_inst)
            
            label_ids = converted.unique()[converted.unique() != 0]
            if len(label_ids) == 0:
                continue
            else:
                # Else pick highest scoring
                scores = [(retrieve_conf_score(pred_inst[0], label), label) for label in label_ids]
                scores.sort(key=lambda x:x[1])

                label_id = scores[0][1].item()

                if scores[0][0] >= conf_thresh and converted[converted != 0].sum() > pixel_thresh:
                    predictions.append({"pos": (x,y), "mask": converted, "score": scores[0][0]})
        elif task == "semantic":
            pred_mask = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256) for _  in range(outputs.masks_queries_logits.shape[0])]))

            if pred_mask[pred_mask != 0].sum() > pixel_thresh:
                predictions.append({"pos": (x,y), "mask": pred_mask})

    print(f"{len(predictions)} objects detected")

    # Filter by confidence score
    return predictions

def process_image(img_path, model, processor, step_size= 4, conf_thresh=0.5, iou_thresh=0.5):
    """
        Process the image with the fine-tuned model in instance segmentation mode.
        A sliding window with size (256,256) is used to extract bb and seg_maps from the image.
        These predictions are then filtered by the confidence score.
        Finally a nms steps is performed to discard similar predictions.
    """
    
    predictions = run_sliding_windows_over_image(img_path, model, processor, step_size, conf_thresh)
    return nms(predictions, iou_thresh)

def bb_intersection_over_union(boxA, boxB):
    """
        Box input is (x,y,w,h)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    y_bot = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, (x_right - x_left)) * max(0, (y_bot - y_top))
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def retrieve_conf_score(pred_inst, label_id):
    for s in pred_inst["segments_info"]:
        if s["label_id"] == label_id:
            return s["score"]

def segments_to_label(predicted):
    w = predicted[0]["segmentation"].shape[0]
    h = predicted[0]["segmentation"].shape[1]
    converted = torch.zeros((len(predicted), w,h))
    for index, el in enumerate(predicted):
        for segment in el["segments_info"]:
            converted[index, el["segmentation"] == segment["id"]] = segment["label_id"]
    return converted

def collect_predictions(dataset, model, processor):
    batch_size = 4
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to("cuda")
    inference_mode(model)

    # background excluded
    per_class_pred = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }

    print(f"Using task {dataset.task}")
    for batch_idx, batch in enumerate(test_dataloader):
        gt_mask = dataset.get_gt_mask(batch)

        batch.pop("mask_labels", None)
        batch = {k:v.to("cuda") for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        pred_inst = processor.post_process_instance_segmentation(outputs, overlap_mask_area_threshold=0.5, target_sizes=[(256,256) for _ in range(outputs.masks_queries_logits.shape[0])])
        converted = segments_to_label(pred_inst)
        
        for i in range(batch_size):
            # Find the label of the highest scoring result, excluding the background class
            label_ids = converted[i].unique()[converted[i].unique() != 0]
            if len(label_ids) == 0:
                # Set to empty result if nothing was found
                label_id = gt_mask[i][gt_mask[i]!=0].unique().item()
                per_class_pred[label_id].append({"segmentation": torch.zeros((256, 256)),
                                            "gt_mask": gt_mask[i],
                                            "conf": 0.0})
            else:
                # Else pick highest scoring
                scores = [(retrieve_conf_score(pred_inst[i], label), label) for label in label_ids]
                scores.sort(key=lambda x:x[1])

                label_id = scores[0][1].item()
                per_class_pred[label_id].append({"segmentation": converted[i],
                                            "gt_mask": gt_mask[i],
                                            "conf": retrieve_conf_score(pred_inst[i], label_id)})
    
    return per_class_pred

def compute_ap(predictions, threshold):
    # Compute precision and recall for each class
    precisions = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }
    recalls = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }

    for cls_id, seg_list in predictions.items():
        tp, fp = 0, 0
        for el in seg_list:
            iou = compute_iou(el["segmentation"].unsqueeze(0), el["gt_mask"].unsqueeze(0), torch.tensor(cls_id))
            tp += torch.sum(iou >= threshold).item()
            fp += torch.sum(iou < threshold).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(predictions[cls_id])

            precisions[cls_id].append(precision)
            recalls[cls_id].append(recall)
            # print(f"{cls_id} | {iou}")

    # compute average precision for each class
    per_class_ap = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }

    for cls_id in per_class_ap.keys():
        recall_levels = [i / 10 for i in range(11)]
        interpolated_precisions = []
        
        for recall_level in recall_levels:
            # Find maximum precision for recall >= recall_level
            precisions_at_recall = [p for r, p in zip(recalls[cls_id], precisions[cls_id]) if r >= recall_level]
            if precisions_at_recall:
                max_precision = max(precisions_at_recall)
            else:
                max_precision = 0.0  # If no recall meets the threshold, precision is 0
            interpolated_precisions.append(max_precision)
        
        # Compute average of the interpolated precisions
        per_class_ap[cls_id] = sum(interpolated_precisions) / len(recall_levels)   

    return per_class_ap


def compute_meanap(model, processor, dataset, verbose=False):
    prev_task = dataset.task
    dataset.set_task("instance")
    # Run the model over the dataset and collect all the predictions divided by class
    per_class_pred = collect_predictions(dataset, model, processor)

    dataset.set_task(prev_task)

    # Sort inplace each class dictionary by confidence score
    for k, v in per_class_pred.items():
        v.sort(key=lambda x:x["conf"], reverse=True)

    cumulated_ap = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }

    # meanAP over different threshold levels
    threshold_levels = np.arange(0.5, 1, 0.05)
    if verbose:
        print("Threshold | per_class_AP")
    for threshold in threshold_levels:
        per_class_ap = compute_ap(per_class_pred, threshold)
        if verbose:
            print(f"{threshold:.2f} | {per_class_ap}")
        for cls_id in cumulated_ap.keys():
            cumulated_ap[cls_id] += per_class_ap[cls_id]

    # compute mean over thresholds
    for cls_id in cumulated_ap.keys():
        cumulated_ap[cls_id] = cumulated_ap[cls_id] / len(threshold_levels)

    if verbose:
        print("\nper_class_AP over thresholds")
        print(cumulated_ap)
    # compute final mAP
    mAP = sum(cumulated_ap.values()) / len(cumulated_ap) 

    return np.mean(mAP)

def compute_accuracy(pred_mask, class_label):
    pred_classes = []

    for i in range(pred_mask.shape[0]):
        # Extract unique classes from the mask
        unique_classes = torch.unique(pred_mask[i])
        pred_classes.append(unique_classes[unique_classes!=0])

    running_acc = 0
    for pred, gt in zip(pred_classes, class_label[class_label != 0]):
        running_acc += (pred == gt).sum()
    return running_acc

def compute_metrics(dataset, model, processor, num_classes, device="cuda"):
    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    model.to(device)
    inference_mode(model)

    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes).to("cuda")

    prev_task = dataset.task
    dataset.set_task("semantic")

    print(f"Using task {dataset.task}")
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
        running_acc += compute_accuracy(pred_mask.cpu(), batch["class_labels"].cpu()).sum()

        pred_classes = []
        for i in range(pred_mask.shape[0]):
            # Extract unique classes from the mask
            unique_classes = torch.unique(pred_mask[i])
            if unique_classes.shape[0] == 1:
                # means there is only the background class
                pred_classes.append(unique_classes)
            else:
                pred_classes.append(unique_classes[unique_classes!=0])
        

        for t, p in zip(batch["class_labels"][batch["class_labels"]!=0].view(-1), pred_classes):
            if p.shape[0] > 1:
                confusion_matrix[t.long(), p.long()] += (p == t).cpu().sum()
            else:
                confusion_matrix[t.long().item(), p.long().item()] += 1

    dataset.set_task(prev_task)

    return running_iou, running_dice, running_acc, confusion_matrix


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

def run_model_on_image(model, processor, path, conditioning):
    image = skimage.io.imread(path)

    mean = image.mean(axis=(0,1))
    std = image.std(axis=(0,1))

    # Image dimensions
    img_height, img_width = image.shape[:2]


    predictions = []

    model.to("cuda")
    inference_mode(model)

    encoded_inputs = processor(
        images=image,
        task_inputs=[conditioning],
        return_tensors="pt",
        image_mean=mean, 
        image_std=std
    )    

    input = {k:v.to("cuda") for k,v in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**input)
    
    if conditioning == "semantic":
        return torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:2] for _ in range(outputs.masks_queries_logits.shape[0])]))
    elif conditioning == "instance":
        pred_inst = processor.post_process_instance_segmentation(outputs, overlap_mask_area_threshold=0.5, target_sizes=[image.shape[:2] for _ in range(outputs.masks_queries_logits.shape[0])])
        converted = segments_to_label(pred_inst)
        return converted

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