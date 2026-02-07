import pandas as pd
import torch
import os
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

seed_value = 69  # You can use any integer value for the seed

# Set the seed for Python's built-in random library
random.seed(seed_value)

# Set the seed for NumPy
#np.random.seed(seed_value)

# Set the seed for PyTorch on CPU and GPU (if available)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If you are using multiple GPUs

# Ensure deterministic behavior (optional but recommended)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# Load the training and validation datasets
train_df = pd.read_csv("data/train.csv")  # Replace with your actual file path
val_df = pd.read_csv("data/val.csv")  # Replace with your actual file path

# Convert the 'label_vector' from string representation to an actual list
train_df['label_vector'] = train_df['label_vector'].apply(ast.literal_eval)
val_df['label_vector'] = val_df['label_vector'].apply(ast.literal_eval)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
print(len(train_df['label_vector'].iloc[0]))
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", num_labels=len(train_df['label_vector'].iloc[0]), ignore_mismatched_sizes=True)

# Tokenize the input text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Convert labels to the appropriate format
def convert_labels_to_float(examples):
    examples['labels'] = [torch.tensor(label_vector, dtype=torch.float32) for label_vector in examples['label_vector']]
    return examples

train_dataset = train_dataset.map(convert_labels_to_float, batched=True)
val_dataset = val_dataset.map(convert_labels_to_float, batched=True)

# Set the format for PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# List to store F1 scores and corresponding epochs
f1_scores = []

# Define the training arguments with early stopping
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,  # Load the best model found during training at the end
    metric_for_best_model="f1",  # Specify the metric to monitor for early stopping
    greater_is_better=True,  # Specify whether the higher metric is better
)

# Define a function to compute metrics and store F1 score
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='micro')  # 'macro' for multilabel
    acc = accuracy_score(p.label_ids, preds)
    f1_scores.append(f1)  # Save F1 score for the epoch
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Create the Trainer object with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping callback
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_roberta_multilabel_early")
tokenizer.save_pretrained("./trained_roberta_multilabel_early")

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print(f"Validation Results: {results}")

# Save the F1 scores to a CSV file
f1_scores_df = pd.DataFrame(f1_scores)
f1_scores_df.to_csv("f1_scores_per_epoch.csv", index=False)

print(f"F1 scores have been saved to 'f1_scores_per_epoch.csv'.")
