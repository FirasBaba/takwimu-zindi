import argparse
import os
import warnings

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import MT5Tokenizer

from working import config
from working.models import mT5Translator

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--beam', type=int, default=None, help='num beam depth')
parser.add_argument('--pn', type=str, default=None, help='Post name of the csv file')
args = parser.parse_args()

model = mT5Translator(config.model_name_or_path, from_tf=True)

model.load_state_dict(torch.load("weights/first_11__.pth"))
tokenizer = MT5Tokenizer.from_pretrained(config.tokenizer_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

data = pd.read_csv("input/Test.csv")
sample = pd.read_csv("input/SampleSubmission.csv")["ID"]

model.eval()

final_score = 0
final_answer = []


batch_size = 48
n_steps = int(np.ceil(len(data[["French"]].values)/batch_size))
tqdm_bar = tqdm((range(n_steps)))


for i in tqdm_bar:
    fr_sents =  data[["French"]].values[i*batch_size:(i+1)*batch_size]
    tr_langs =  data[["Target_Language"]].values[i*batch_size:(i+1)*batch_size]
    text = [f"translate French to {tr_lang[0]}: " + str(fr_sent[0]) + " </s>"
            for tr_lang, fr_sent in zip(tr_langs,fr_sents)]

    max_len = 100

    encoding = tokenizer.batch_encode_plus(text,padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.translator.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=max_len,
        top_k=120,
        top_p=0.998,
        early_stopping=True,
        num_return_sequences=1,
        num_beams= args.beam,
    )

    sentences_batch = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    final_answer.extend(sentences_batch)


data["Target"] = final_answer
sample = pd.merge(sample, data[["ID", "Target"]], on="ID", how="left")

n = len(os.listdir("submissions/"))
sample.to_csv(f"submissions/submit_{n}_{args.pn}.csv", index=False)
