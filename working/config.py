data_dir = "input/"
model_name_or_path = "google/mt5-small"
tokenizer_name_or_path = "google/mt5-small"


max_seq_length_french = 100
max_seq_length_target = 100
learning_rate = 8e-4
weight_decay = 0.0
adam_epsilon = 1e-8
warmup_steps = 0
train_batch_size = 24
eval_batch_size = 24
n_epochs = 12
gradient_accumulation_steps = 1
n_workers = 4

validation = True