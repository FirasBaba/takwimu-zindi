import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers.utils.dummy_tf_objects import TF_MODEL_FOR_PRETRAINING_MAPPING

training_data = pd.read_csv("input/Train.csv")

def lower(text):
    return text.strip()

f = open("input/jw300_ee.fr", "r")
french_sentences = f.readlines()
f.close()
f = open("input/jw300.ee", "r")
ee_sentences = f.readlines()
f.close()

id_column = [1] * len(french_sentences)
french_sentences = list(map(lower, french_sentences))
ee_sentences = list(map(lower, ee_sentences))
external_data_ee = pd.DataFrame(columns=["ID","French","Target"])
external_data_ee["ID"] = id_column
external_data_ee["French"] = french_sentences
external_data_ee["Target"] = ee_sentences
external_data_ee["Target_Language"] = "Ewe"

f = open("input/jw300_fon.fr", "r")
french_sentences = f.readlines()
f.close()
f = open("input/jw300.fon", "r")
fon_sentences = f.readlines()
f.close()

id_column = [1] * len(french_sentences)
french_sentences = list(map(lower, french_sentences))
fon_sentences = list(map(lower, fon_sentences))
external_data_fon = pd.DataFrame(columns=["ID","French","Target"])
external_data_fon["ID"] = id_column
external_data_fon["French"] = french_sentences
external_data_fon["Target"] = fon_sentences
external_data_fon["Target_Language"] = "Fon"

external_data = pd.concat([external_data_ee, external_data_fon]).sample(frac=1.0).reset_index(drop=True)
external_data.to_csv("input/external_jw300_data.csv", index=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
for fold, (train_indx, val_indx) in enumerate(skf.split(training_data, training_data.Target_Language)):
    train_f = training_data.loc[train_indx].reset_index(drop=True)
    val_f = training_data.loc[val_indx].reset_index(drop=True)

    train_f.to_csv(f"input/folds/train_fold_{fold}.csv")
    val_f.to_csv(f"input/folds/validation_fold_{fold}.csv")
