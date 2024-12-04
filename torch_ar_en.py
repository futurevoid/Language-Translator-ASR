import torch
from transformers import AutoTokenizer, MarianConfig
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch.nn as nn
from torch.optim import AdamW
import os

# Define the Transformer model for translation
class TranslationModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            nhead=num_heads,
            batch_first=True,
        )
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding the input
        src_emb = self.embedding(input_ids)
        tgt_emb = src_emb  # Decoder uses the same embeddings in this example

        # Use transformer to process the embeddings
        output = self.transformer(src_emb, tgt_emb)
        logits = self.fc_out(output)

        if labels is not None:
            # Calculate loss if labels are provided (during training)
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Use tokenizer's pad_token_id
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

        return logits

# Hyperparameters
hidden_size = 256
num_layers = 4
num_heads = 8
batch_size = 16
num_epochs = 3
learning_rate = 5e-4
max_seq_length = 128

print("Loading the dataset...")
# Load the dataset
dataset = load_dataset("MagedSaeed/opus-100_ar_en_experimental")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print("Loading the tokenizer...")
# Load the tokenizer
tokenizer_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print("Preprocessing the dataset...")
# Preprocess the dataset
def preprocess_function(examples):
    inputs = tokenizer(examples["ar"], truncation=True, padding="max_length", max_length=max_seq_length)
    targets = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=max_seq_length)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_data = train_dataset.map(preprocess_function, batched=True)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model
    evaluation_strategy="no",  # Disable evaluation during training
    save_strategy="steps",  # Save model at regular intervals
    per_device_train_batch_size=8,  # Smaller batch size to fit in memory
    per_device_eval_batch_size=8,  # Evaluation batch size
    num_train_epochs=5,  # Number of training epochs
    save_total_limit=2,  # Keep only the last 2 saved models
    fp16=True,  # Mixed precision training for faster performance
    predict_with_generate=True,  # Use generation for predictions
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
    remove_unused_columns=False,  # Prevent removing unused columns
)

# Initialize the model
vocab_size = tokenizer.vocab_size
model = TranslationModel(vocab_size, hidden_size, num_layers, num_heads)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)

print("Initializing the Trainer...")
# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

print("Starting the training process...")
# Start training
trainer.train()

# Save the model and tokenizer
trainer.save_model("./translation_model")  # Saves the model weights
tokenizer.save_pretrained("./translation_model")  # Saves the tokenizer

print("Model, tokenizer, and training arguments saved using Trainer.")
