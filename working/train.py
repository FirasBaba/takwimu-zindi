import argparse
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, MT5Tokenizer, get_linear_schedule_with_warmup

from rouge_score import rouge_scorer

from . import config
from .dataset import TakwimuDataset
from .models import mT5Translator

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Training fold')
args = parser.parse_args()

TRAIN_PATH = f"input/folds/train_fold_{args.fold}.csv"
VAL_PATH = f"input/folds/validation_fold_{args.fold}.csv"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(2021)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MT5Tokenizer.from_pretrained(config.tokenizer_name_or_path)
model = mT5Translator(config.model_name_or_path, from_tf=True)

train_ds = TakwimuDataset(
        tokenizer=tokenizer,
        csv_path=TRAIN_PATH,
        max_len_french=config.max_seq_length_french,
        max_len_target=config.max_seq_length_target,
        task="train"
    )
train_dl = DataLoader(
            train_ds, batch_size=config.train_batch_size, num_workers=config.n_workers
        )

validation_ds = TakwimuDataset(
        tokenizer=tokenizer,
        csv_path=VAL_PATH,
        max_len_french=config.max_seq_length_french,
        max_len_target=config.max_seq_length_target,
        task="validation"
    )
validation_dl = DataLoader(
            validation_ds, batch_size=config.eval_batch_size, num_workers=config.n_workers
        )


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": config.weight_decay,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

total_steps = len(train_ds) // (config.train_batch_size)// config.gradient_accumulation_steps * float(config.n_epochs)

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=config.learning_rate,
    eps=config.adam_epsilon,
)
scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

# Training loop
model.zero_grad()
model.to(device)
best_loss = 9999
for counter in range(config.n_epochs):

    epoch_iterator_train = tqdm(train_dl)
    tr_loss = 0.0
    for step, batch in enumerate(epoch_iterator_train):
        model.train()
        source_ids, source_mask = batch["source_ids"].to(device), batch["source_mask"].to(device)
        target_ids, target_mask = batch["target_ids"].to(device), batch["target_mask"].to(device)

        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=source_ids,
            attention_mask=source_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_mask)

        loss = outputs[0]


        (loss).backward()
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        tr_loss += loss.item()
        epoch_iterator_train.set_postfix(batch_loss=(loss.item()), loss=(tr_loss/(step+1)))

    print(f"EPOCH {counter}/{config.n_epochs}: Training average loss: {tr_loss/(step+1)}")

    if config.validation:
        print("Predicting on the validation set")
        val_loss = 0.0
        epoch_iterator_val = tqdm(validation_dl)
        for  step, batch in enumerate(epoch_iterator_val):
            model.eval()
            source_ids, source_mask = batch["source_ids"].to(device), batch["source_mask"].to(device)
            target_ids, target_mask = batch["target_ids"].to(device), batch["target_mask"].to(device)

            target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    
            outputs = model(input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_mask)
            loss = outputs[0]
            val_loss += loss.item()
            epoch_iterator_val.set_postfix(batch_loss=(loss.item()), loss=(val_loss/(step+1)))
        print(f"EPOCH {counter}/{config.n_epochs}: Validation average loss: {val_loss/(step+1)}")
        if (val_loss/(step+1)) < best_loss:
            print("Saving the Translator...")
            best_loss = val_loss/(step+1)
            torch.save(model.state_dict(), f"weights/first_{counter}__.pth")
        else:
            print("Saving the Translator BAAAAAAAAAD...")
            torch.save(model.state_dict(), f"weights/first_{counter}__.pth")

    else:
        print("Saving the Translator...")
        torch.save(model.state_dict(), "weights/first_.pth")
