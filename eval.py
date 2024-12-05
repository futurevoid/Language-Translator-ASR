import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

# Load the saved pre-trained model and tokenizer
print("Loading the saved model and tokenizer...")
saved_model_path = "./translation_model"  # Path to your saved model
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

# Function to evaluate and get the loss
def evaluate_model():
    eval_results = trainer.evaluate()
    return eval_results

# Load the BLEU metric
bleu = evaluate.load("sacrebleu")

# Convert token IDs to text
def decode_sequences(pred_ids, label_ids):
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return preds, labels

# Store predictions and references for BLEU
all_preds = []
all_labels = []

# Function to calculate BLEU score
def compute_bleu():
    all_preds.clear()  # Clear previous predictions
    all_labels.clear()  # Clear previous labels
    for batch in trainer.get_eval_dataloader():
        input_ids = batch['input_ids']
        labels = batch['labels']

        with torch.no_grad():
            # Perform generation
            generated_ids = model.generate(input_ids)

        # Decode predictions and labels
        preds, refs = decode_sequences(generated_ids, labels)

        # Append predictions and references
        all_preds.extend(preds)
        all_labels.extend([[ref] for ref in refs])  # SacreBLEU expects a list of references per sample

    # Calculate BLEU score
    bleu_score = bleu.compute(predictions=all_preds, references=all_labels)
    return bleu_score['score']

# Track BLEU score over epochs
bleu_scores = []

# Simulate training with multiple epochs (if you're running actual epochs)
epochs = 3  # Set to the number of epochs you're running

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} - Calculating BLEU score...")
    bleu_score = compute_bleu()
    bleu_scores.append(bleu_score)
    print(f"BLEU Score for epoch {epoch + 1}: {bleu_score:.2f}")

# Plot BLEU score over epochs
plt.plot(range(1, epochs + 1), bleu_scores, label="BLEU Score", marker='o')
plt.xlabel("Epochs")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Epochs")
plt.legend()
plt.show()

# Track eval loss over epochs (assuming you call evaluate after each epoch)
eval_losses = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} - Evaluating...")
    eval_results = evaluate_model()
    eval_losses.append(eval_results['eval_loss'])
    print(f"Evaluation Loss for epoch {epoch + 1}: {eval_results['eval_loss']:.4f}")

# Plot learning curve (Loss vs. Epochs)
plt.plot(range(1, epochs + 1), eval_losses, label="Eval Loss", marker='o')
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
    for pred, label in zip(preds, labels):
        pred_tokens = pred[1:-1]  # Exclude <s> and </s> tokens if any
        label_tokens = label[1:-1]  # Exclude padding tokens
        
        # Compare token-by-token
        correct += (pred_tokens == label_tokens).sum().item()
        total += len(label_tokens)

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
