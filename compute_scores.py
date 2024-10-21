import matplotlib.pyplot as plt
import sys
import argparse
import logging

sys.path.append('../src')

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, OneFormerConfig
from transformers.models.oneformer.image_processing_oneformer import load_metadata, prepare_metadata
from safetensors.torch import load_model, save_model
from ignite.metrics import *
from src.config import DATASET_DIR, CLASS_INFO_PATH, MAIN_PROJECT_DIR, GALAXY_MEAN, GALAXY_STD
from src.visual import *
from src.dataset import *
from src.utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for computing metrics over a trained model")

    parser.add_argument(
        '-f', 
        '--file_name', 
        required=True,  
        help="The path of the .safetensors you want to load"
    )

    args = parser.parse_args()

    return args

def main(safetensors_path):
    model_card = "shi-labs/oneformer_ade20k_swin_tiny"

    id2label, label2id = get_id2label_mappings()
    config = OneFormerConfig.from_pretrained(model_card, 
                                            num_classes = len(id2label),
                                            id2label = id2label,
                                            label2id = label2id,
                                            is_training=False)

    model = OneFormerForUniversalSegmentation.from_pretrained(model_card, config=config, ignore_mismatched_sizes=True)
    processor = OneFormerProcessor.from_pretrained(model_card)

    # Metadata must be set according to the dataset through the class_info.json file. Background class must be specified as well. 
    processor.image_processor.repo_path = MAIN_PROJECT_DIR
    processor.image_processor.class_info_file = os.path.join(MAIN_PROJECT_DIR, CLASS_INFO_PATH)
    processor.image_processor.metadata = prepare_metadata(load_metadata(MAIN_PROJECT_DIR, os.path.join(MAIN_PROJECT_DIR, CLASS_INFO_PATH)))
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
    processor.image_processor.do_resize = False
    processor.image_processor.do_rescale = False

    # strict = False because we don't need the text mapper at inference time
    # unmatched = load_model(model, "../safetensors/galaxy_sem_5_1.safetensors", strict=False)
    unmatched = load_model(model, safetensors_path, strict=False)

    # Only the text_mapper should be unmatched with the pretrained model
    for el in unmatched[1]:
        if el.split(".")[1] != "text_mapper":
            print(f"BAD: {el}")

    dataset_test = GalaxyDataset(DATASET_DIR, "test", processor, CLASS_INFO_PATH, None, GALAXY_MEAN, GALAXY_STD, load_on_demand=True)

    run_iou, run_dice, run_acc, confusion_matrix = compute_metrics(dataset_test, model, processor, len(id2label))
    print(f"mIoU: {run_iou / len(dataset_test):.4f}")
    print(f"Dice: {run_dice / len(dataset_test):.4f}")
    print(f"Accuracy: {run_acc / len(dataset_test):.4f}")

    map = compute_meanap(model, processor, dataset_test, verbose=True)
    print(f"AP (at IoU=.50:.05:.95) = {map:.4f}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    args = parse_arguments()

    main(args.file_name)