import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import inference_mode, segments_to_label
# from src.utils import inference_mode, segments_to_label
from torch.utils.data import DataLoader
from matplotlib.patches import Patch

def plot_distribution(distributions, labels):
    fig, axs = plt.subplots(1, len(distributions))
    fig.set_size_inches(20, 5)
    for distribution, label, axis in zip(distributions, labels, axs):
        sorted_distr = distribution.most_common()
        key_labels = [el[0] for el in sorted_distr]
        values = [el[1] for el in sorted_distr]
        axis.bar(key_labels, values)
        axis.set_title(label)
    plt.ylabel('Frequency')
    plt.show()

def display_n_samples(dataset, n, from_dir=True):
    """
    Call display_top_masks over n images
    Args:
        dataset: 
            the dataset from which the image will be picked
        n (int): 
            how many samples to show
        from_dir (bool): 
            wether loading the image from the real directory or directly use the dataset output
    """
    if from_dir:
        images = np.random.choice(dataset.image_info, n)
        for image_obj in images:
            image = dataset.load_image(image_obj)
            mask = dataset.load_mask(image_obj)
            class_ids = np.array([int(x) for x in image_obj["class_ids"]])
            display_top_masks(image, mask, class_ids, dataset.class_names)
    else:
        idxs = np.random.choice(np.arange(len(dataset)), 3)
        for idx in idxs:
            el = dataset[idx]
            display_images([el["pixel_values"].permute((1,2,0)), el["mask_labels"][1]], None, 2)
        

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(6, 6 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        if isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
        plt.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
    
def display_top_masks(image, mask, class_ids, class_names, limit=1):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))

    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")

def mask_gt_overlap(gt_mask, pred_mask, id2label):
    rgb = torch.zeros((3, 256, 256))
    rgb[1, :,:] = torch.logical_and(gt_mask, pred_mask.cpu())
    rgb[0, :,:] = torch.logical_and(gt_mask, torch.logical_not(pred_mask.cpu()))
    rgb[2, :,:] = torch.logical_and(pred_mask.cpu(), torch.logical_not(gt_mask))

    # plt.figure(figsize=(6, 6))
    # plt.imshow(rgb.permute(1,2,0))
    # plt.title('Overlay of Ground Truth and Prediction Masks')
    
    gt_class = id2label[gt_mask[gt_mask != 0].unique().item()]
    if len(pred_mask.unique()) > 1:
        pred_class = id2label[pred_mask[pred_mask != 0].unique().item()]
    else:
        pred_class = None

    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='Ground Truth Only'),
        Patch(facecolor='blue', edgecolor='b', label='Prediction Only'),
        Patch(facecolor='green', edgecolor='g', label='Overlap'),
        Patch(fill=False, edgecolor='none', label=f'GT: {gt_class}'),
        Patch(fill=False, edgecolor='none', label=f'Predicted: {pred_class}')
    ]

    return rgb.permute(1,2,0), legend_elements
    # plt.legend(handles=legend_elements, loc='upper right', fontsize='x-small')
    # plt.axis('off')  
    # plt.show()

def random_get_instance_masks(dataset, model, processor, id2label):
    prev_task = dataset.task
    dataset.set_task("instance")
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.to("cuda")
    inference_mode(model)

    print(f"Using task {dataset.task}")
    for batch_idx, batch in enumerate(test_dataloader):
        gt_mask = dataset.get_gt_mask(batch)

        batch.pop("mask_labels", None)
        batch = {k:v.to("cuda") for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        pred_inst = processor.post_process_instance_segmentation(outputs, overlap_mask_area_threshold=0.5, target_sizes=[(256,256) for _ in range(outputs.masks_queries_logits.shape[0])])
        converted = segments_to_label(pred_inst)
        break

    mask_gt_overlap(gt_mask, converted, id2label)
    dataset.set_task(prev_task)

def show_confusion_matrix(confusion_matrix, id2label):
    # Convert tensor to numpy array for processing
    confusion_matrix = confusion_matrix.cpu().numpy()

    # Normalize the confusion matrix per class (row-wise normalization)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums

    # Create labels for the axes
    labels = [id2label[i] for i in range(len(id2label))]

    # Plot the normalized confusion matrix with color intensity based on the normalized values
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(normalized_confusion_matrix, cmap='Blues')

    # Add a color bar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Label axes with the class names
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)

    # Add both total count and normalized percentage in each cell
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            count = confusion_matrix[i, j]  # Raw count
            percentage = normalized_confusion_matrix[i, j] * 100  # Percentage
            if count > 0:
                ax.text(j, i, f'{int(count)}\n({percentage:.1f}%)', va='center', ha='center', fontsize=10)

    # Set axis labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.title('Confusion Matrix with Counts and Percentages')
    plt.tight_layout()
    plt.show()

def compare_models(model_1, model_2, processor, dataset_test, id2label, conditioning, model_names=("model1", "model2")):
    bs=4
    test_dataloader = DataLoader(dataset_test, batch_size=bs, shuffle=True)

    model_1.to("cuda")
    model_2.to("cuda")
    inference_mode(model_1)
    inference_mode(model_2)

    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0

    dataset_test.set_task(conditioning)

    for batch_idx, batch in enumerate(test_dataloader):
        # print(f"Iteration n. {batch_idx+1} / {len(test_dataloader)}", end="\r")
        gt_mask = dataset_test.get_gt_mask(batch)

        batch.pop("mask_labels", None)
        batch.pop("text_inputs", None)
        batch = {k:v.to("cuda") for k,v in batch.items()}
        with torch.no_grad():
            outputs_seg = model_1(**batch)
            outputs_seg_inst = model_2(**batch)

        # pred_mask = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256) for _  in range(outputs.masks_queries_logits.shape[0])]))
        pred_inst_1 = segments_to_label(processor.post_process_instance_segmentation(outputs_seg, overlap_mask_area_threshold=0.5, target_sizes=[(256,256) for _ in range(outputs_seg.masks_queries_logits.shape[0])]))
        pred_inst_2 = segments_to_label(processor.post_process_instance_segmentation(outputs_seg_inst, overlap_mask_area_threshold=0.5, target_sizes=[(256,256) for _ in range(outputs_seg_inst.masks_queries_logits.shape[0])]))
        break
    
    # Plot comparison
    fig, axs = plt.subplots(2, bs, figsize=(17, 6))

    axs[0, 0].set_ylabel(model_names[0])
    axs[1, 0].set_ylabel(model_names[1])

    for i in range(bs):
        overlap_1, leg_1 = mask_gt_overlap(gt_mask[i], pred_inst_1[i], id2label)
        overlap_2, leg_2 = mask_gt_overlap(gt_mask[i], pred_inst_2[i], id2label)
        
        axs[0, i].imshow(overlap_1)
        axs[0, i].legend(handles=leg_1, loc='upper right', fontsize='xx-small')

        axs[1, i].imshow(overlap_2)
        axs[1, i].legend(handles=leg_2, loc='upper right', fontsize='xx-small')

    fig.suptitle(f'{conditioning} conditioning', fontsize=16)
