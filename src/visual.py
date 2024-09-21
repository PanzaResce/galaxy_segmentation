import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

def plot_distribution(distributions, labels):
    fig, axs = plt.subplots(1, len(distributions))
    fig.set_size_inches(15, 5)
    for distribution, label, axis in zip(distributions, labels, axs):
        key_labels = [el.split(" ")[-1] for el in distribution.keys()]
        axis.bar(key_labels, distribution.values())
        axis.set_title(label)
    plt.xlabel('Class')
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

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb.permute(1,2,0))
    plt.title('Overlay of Ground Truth and Prediction Masks')
    
    gt_class = id2label[gt_mask[gt_mask != 0].unique().item()]
    pred_class = id2label[pred_mask[pred_mask != 0].unique().item()]

    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='Ground Truth Only'),
        Patch(facecolor='blue', edgecolor='b', label='Prediction Only'),
        Patch(facecolor='green', edgecolor='g', label='Overlap'),
        Patch(fill=False, edgecolor='none', label=f'GT: {gt_class}'),
        Patch(fill=False, edgecolor='none', label=f'Predicted: {pred_class}')
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize='x-small')
    plt.axis('off')  
    plt.show()