import os
import torch
import json
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import multiprocessing as mp
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    def __init__(self, dataset_dir, subset, processor, cls_mapping_path, transform, mean, std):
        self.image_info = []
        self.image_data = []
        self.mask_data = []
        self.dataset_classes = []

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.processor = processor
        self.class_mapping = self.load_cls_mapping(cls_mapping_path)
        self.transform = transform
        self.mean = mean
        self.std = std
        self.load_galaxia(dataset_dir, subset)
    
    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        mask = self.mask_data[idx]
        
        # Prepare task inputs
        task_inputs = ["semantic"]
        
        # Apply transform augmentation
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

        # Apply the processor to the image and masks
        encoded_inputs = self.processor(
            images=transformed_image,
            task_inputs=task_inputs,
            segmentation_maps=np.squeeze(transformed_mask),     # squeeze is necessary because the image processor expect ndims=2
            return_tensors="pt",
            image_mean=self.mean,
            image_std=self.std
        )
        
        # Extract the required outputs
        # binary_masks = encoded_inputs["pixel_values"]
        # labels = torch.tensor(class_ids, dtype=torch.int64)
        # text_inputs = encoded_inputs["text_inputs"]
        # task_inputs = encoded_inputs["task_inputs"]
        
        # return binary_masks, labels, text_inputs, task_inputs
        inputs = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in encoded_inputs.items()}
        # self.type_conversion(inputs)
        return inputs

    def get_unorm_image(self, idx):
        image = self.__getitem__(idx)["pixel_values"]
        original_image = skimage.io.imread(self.image_info[idx]["path"])
        image_average = original_image.mean(axis=(0,1))
        image_std = original_image.std(axis=(0,1))
        unnormalized_image = (image.numpy() * np.array(image_std)[:, None, None]) + np.array(image_average)[:, None, None]
        return np.moveaxis(unnormalized_image, 0, -1).astype(np.uint8)

    @property
    def class_names(self):
        return list(self.class_mapping.values())

    @property
    def reverse_class_mapping(self):
        """From names to ids"""
        return {y: x for x, y in self.class_mapping.items()}

    def load_cls_mapping(self, cls_mapping_path):
        with open(cls_mapping_path) as fp:
            d = json.load(fp)

        class_mapping = {k:v["name"] for k,v in d.items()}
        assert class_mapping != {}
        return class_mapping

    def load_galaxia(self, dataset_dir, subset):
        """Load a subset of the galaxyzoo dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, val or test
        """
        # Add classes
        for cls, id in self.class_mapping.items():
            self.add_class("galaxia", id, cls)

        assert subset in ["train", "val", "test"]

        dataset = json.load(open(os.path.join(dataset_dir, "galaxy_"+subset+".json")))
        dataset = list(dataset.values())

        # Skip unannotated images
        dataset = [a for a in dataset if a['regions']]

        # Add images
        for galxy_obj in dataset:
            
            # The dataset can have more than just one object per image
            if type(galxy_obj['regions']) is dict:
                polygons = [r['shape_attributes'] for r in galxy_obj['regions'].values()]
                objects = [s['region_attributes'] for s in galxy_obj['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in galxy_obj['regions']]
                objects = [s['region_attributes'] for s in galxy_obj['regions']]
                        
            class_ids = []
            
            # Add the ids according to the class_mapping
            for obj in objects:
                class_ids.append(self.reverse_class_mapping[obj["object_name"]])
            
            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(dataset_dir, "original/zoo2Main", galxy_obj['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "galaxia",
                image_id=galxy_obj['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)


    @staticmethod
    def type_conversion(encoded_inputs):
        """Convert statically the dtype of the encoded_inputs fields"""
        encoded_inputs["pixel_values"] = encoded_inputs["pixel_values"].to(torch.float16)
        encoded_inputs["pixel_mask"] = encoded_inputs["pixel_mask"].to(torch.int16)
        encoded_inputs["mask_labels"] = encoded_inputs["mask_labels"].to(torch.float16)
        encoded_inputs["class_labels"] = encoded_inputs["class_labels"].to(torch.int16)


    def load_image(self, image_obj):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(image_obj['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_info):
        """Generate instance masks for an image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with one mask per instance.
        """        
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape [height, width, instance_count]
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to class_id
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = class_ids[i]
            
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "galaxia":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def add_class(self, source, class_id, class_name):
        """Add a new class to the dataset."""
        self.dataset_classes.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })
    
    def add_image(self, source, image_id, path, **kwargs):
        """Add a new image to the dataset."""
        image_info = {
            "source": source,
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.image_data.append(skimage.io.imread(image_info["path"]))
        self.mask_data.append(self.load_mask(image_info))

