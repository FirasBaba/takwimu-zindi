import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers.utils.dummy_tf_objects import TF_MODEL_FOR_PRETRAINING_MAPPING

training_data = pd.read_csv("input/Train.csv")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
for fold, (train_indx, val_indx) in enumerate(skf.split(training_data, training_data.Target_Language)):
    train_f = training_data.loc[train_indx].reset_index(drop=True)
    val_f = training_data.loc[val_indx].reset_index(drop=True)

    train_f.to_csv(f"input/folds/train_fold_{fold}.csv")
    val_f.to_csv(f"input/folds/validation_fold_{fold}.csv")
