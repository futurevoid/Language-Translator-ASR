import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch.nn as nn
import os

# Load the saved pre-trained model and tokenizer
print("Loading the saved model and tokenizer...")
saved_model_path = "./ar_en_model1"  # Path to your saved model
model = MarianMTModel.from_pretrained(saved_model_path)
tokenizer = MarianTokenizer.from_pretrained(saved_model_path)

# Set the model to evaluation mode (important for inference)
model.eval()

# Load the dataset for evaluation (same dataset used in training)
dataset = load_dataset("MagedSaeed/opus-100_ar_en_experimental", split="test")
test_dataset = dataset

# Preprocess the dataset for evaluation
def preprocess_function(examples):
    inputs = tokenizer(examples["ar"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

test_data = test_dataset.map(preprocess_function, batched=True)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Set up the evaluation arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model (though we are not saving here)
    per_device_eval_batch_size=8,  # Batch size for evaluation
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps (you can adjust)
    remove_unused_columns=False,  # Prevent removing unused columns
    predict_with_generate=True,  # Use generation for predictions
)

# Set up the Trainer (only for evaluation, no training)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()

# Print evaluation results (loss, metrics, etc.)
print("Evaluation results:", eval_results)

# Confusion Matrix Calculation
all_preds = []
all_labels = []

# Iterate over the evaluation dataset to get predictions and labels
for batch in trainer.get_eval_dataloader():
    input_ids = batch['input_ids']
    labels = batch['labels']

    with torch.no_grad():
        # Perform generation (not a classification problem)
        generated_ids = model.generate(input_ids)
        preds = generated_ids

    # Store the predictions and labels for confusion matrix
    all_preds.extend(preds.cpu().numpy().flatten())
    all_labels.extend(labels.cpu().numpy().flatten())

# Create confusion matrix (for token-level predictions)
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=tokenizer.convert_ids_to_tokens(range(len(tokenizer))),
            yticklabels=tokenizer.convert_ids_to_tokens(range(len(tokenizer))))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot learning curve (Loss vs. Epochs)
# We will use the eval results from trainer
train_losses = eval_results.get("eval_loss", [])
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Eval Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve (Loss vs Epochs)")
plt.legend()
plt.show()

# Calculate accuracy on test data
correct = 0
total = 0

for batch in trainer.get_eval_dataloader():
    input_ids = batch['input_ids']
    labels = batch['labels']

    with torch.no_grad():
        # Perform generation (not classification)
        generated_ids = model.generate(input_ids)
        preds = generated_ids

    # Compare predictions to true labels (ignoring padding tokens)
    correct += (preds == labels).masked_select(labels != tokenizer.pad_token_id).sum().item()
    total += (labels != tokenizer.pad_token_id).sum().item()

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
