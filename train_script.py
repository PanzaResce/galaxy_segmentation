import logging
import sys
import json
# sys.path.append('../src')

import albumentations as A
import argparse
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, OneFormerConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers.models.oneformer.image_processing_oneformer import load_metadata, prepare_metadata
from safetensors.torch import load_model, save_model
from src.config import DATASET_DIR, CLASS_INFO_PATH, MAIN_PROJECT_DIR, GALAXY_MEAN, GALAXY_STD
from src.visual import *
from src.dataset import *
from src.utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for training a model with recovery option.")

    parser.add_argument(
        '-rec', 
        '--recovery', 
        action='store_true', 
        help="Run the script in recover execution mode if set, read from \"safetensors/galaxy.safetensors\""
    )

    parser.add_argument(
        '-n', 
        type=int, 
        required=True, 
        help="The total number of epochs"
    )

    parser.add_argument(
        '-end', 
        type=int, 
        help="The epoch until which to train the model. Defaults to n_epoch if not provided."
    )

    args = parser.parse_args()

    if args.end is None:
        args.end = args.n

    return args

def main(n_epochs, end_epoch, recover_exec):
    model_card = "shi-labs/oneformer_ade20k_swin_tiny"

    id2label, label2id = get_id2label_mappings()
    config = OneFormerConfig.from_pretrained(model_card, 
                                         num_classes = len(id2label),
                                         id2label = id2label,
                                         label2id = label2id,
                                         is_training=True)
    
    model = OneFormerForUniversalSegmentation.from_pretrained(model_card, config=config, ignore_mismatched_sizes=True)
    processor = OneFormerProcessor.from_pretrained(model_card)

    # Metadata must be set according to the dataset through the class_info.json file. Background class must be specified as well. 
    processor.image_processor.repo_path = MAIN_PROJECT_DIR
    processor.image_processor.class_info_file = os.path.join(MAIN_PROJECT_DIR, CLASS_INFO_PATH)
    processor.image_processor.metadata = prepare_metadata(load_metadata(MAIN_PROJECT_DIR, os.path.join(MAIN_PROJECT_DIR, CLASS_INFO_PATH)))
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
    processor.image_processor.do_resize = False
    processor.image_processor.do_rescale = False

    # Augmentations
    transform = A.Compose([
        A.OneOf([
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.Rotate(limit=(90), p=1),
                    A.Rotate(limit=(180), p=1),
                    A.Rotate(limit=(270), p=1)
                ], p=1),
                # A.MultiplicativeNoise((0.9, 1.1), False, True, p=0.5)
            ]),
            A.NoOp()
        ], p=1.0)
    ])

    print("LOADING DATASET...")
    # dataset_train = GalaxyDataset(DATASET_DIR, "train", processor, CLASS_INFO_PATH, transform, GALAXY_MEAN, GALAXY_STD, load_on_demand=True, subset_idx=20)
    # dataset_val = GalaxyDataset(DATASET_DIR, "val", processor, CLASS_INFO_PATH, transform, GALAXY_MEAN, GALAXY_STD, load_on_demand=True, subset_idx=20)
    dataset_train = GalaxyDataset(DATASET_DIR, "train", processor, CLASS_INFO_PATH, transform, GALAXY_MEAN, GALAXY_STD, load_on_demand=True)
    dataset_val = GalaxyDataset(DATASET_DIR, "val", processor, CLASS_INFO_PATH, transform, GALAXY_MEAN, GALAXY_STD, load_on_demand=True)


    batch_size = 2
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    starting_epoch = 0
    verbose = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_loss = []
    val_loss = []

    # Hyperparameters
    weight_decay = 0.1
    base_lr = 1e-4
    grad_clip = 1


    if recover_exec:
        with open("safetensors/galaxy_train.json") as json_data:
            rec = json.load(json_data)
        last_iteration = rec["last_iteration"]
        starting_epoch = rec["last_epoch"]
        lr = float(rec["last_lr"])
        print(f"Recovering execution from epoch {starting_epoch} | lr={lr}")
        load_model(model, "safetensors/galaxy.safetensors")
    
    model.to(device)

    # Scheduler and Optimizer
    warmup_iters = 10
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    sched = WarmupPolyLR(optimizer, len(dataset_train)*n_epochs, warmup_iters=warmup_iters, warmup_factor=0.01)

    if recover_exec:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        sched.last_epoch = last_iteration
        sched._last_lr = [group['lr'] for group in optimizer.param_groups]

    # print(f"Sched --> last_epoch: {sched.last_epoch} | max_iters: {sched.max_iters} | power: {sched.power} | base_lrs: {sched.base_lrs}")

    print(f"Training in {dataset_train.task} mode")

    for epoch in range(starting_epoch, end_epoch):
        print(f"Epoch {epoch}")
        training_mode(model)


        for batch_idx, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            
            tr_loss.append(outputs.loss.item())

            outputs.loss.backward()
            norm = clip_grad_norm_(model.parameters(), grad_clip)

            # print every 5% of progress
            if (batch_idx + 1) % (len(train_dataloader) // 20) == 0 or batch_idx < 10 or verbose:
                print(f"Loss at iteration n. {(batch_idx + 1)} / {len(train_dataloader)}: {outputs.loss.item():.6f} | lr: {sched.get_last_lr()} | norm: {norm:.4f}")

            # print(f"Last epoch: {sched.last_epoch}")    
            optimizer.step()
            sched.step()
        
        # Validation loop
        evaluation_mode(model)
        val_running_loss = 0.0
        print("Validation loop")
        for batch_idx, batch in enumerate(val_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
            val_running_loss += outputs.loss.item()
        
        val_loss.append(val_running_loss / len(val_dataloader) * batch_size)
        print(f"Validation loss at epoch {epoch}: {val_loss[-1]}")
    
    save_model(model, "safetensors/galaxy.safetensors")

    # print(f"Sched --> last_epoch: {sched.last_epoch} | max_iters: {sched.max_iters} | power: {sched.power} | base_lrs: {sched.base_lrs}")

    # Save json to recover training
    if end_epoch != n_epochs:
        print("Saving json...")
        with open("safetensors/galaxy_train.json", "w") as fp:
            rec = {"last_lr": sched.get_last_lr()[0], "last_iteration": sched.last_epoch, "last_epoch": end_epoch}
            json.dump(rec , fp)
    
    write_mode = "a" if starting_epoch != 0 else "w"
    
    # save loss to file
    with open('out/tr_loss.txt', write_mode) as outfile:
        if write_mode == "a":
            outfile.write("\n")
        outfile.write('\n'.join(str(i) for i in tr_loss))

    with open('out/val_loss.txt', write_mode) as outfile:
        if write_mode == "a":
            outfile.write("\n")
        outfile.write('\n'.join(str(i) for i in val_loss))


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    args = parse_arguments()

    main(args.n, args.end, args.recovery)
    # print(args)


