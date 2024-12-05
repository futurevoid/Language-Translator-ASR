import torch
from transformers import AutoTokenizer, MarianConfig
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
import torch.nn as nn
from torch.optim import AdamW
import evaluate
import matplotlib.pyplot as plt

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
hidden_size = 512
num_layers = 4
num_heads = 8
batch_size = 16
num_epochs = 3
learning_rate = 5e-5
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

eval_data = test_dataset.map(preprocess_function, batched=True)
eval_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="steps",  # Save model at regular intervals
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    save_total_limit=2,  # Keep only the last 2 saved models
    fp16=True,  # Mixed precision training for faster performance
    predict_with_generate=True,  # Use generation for predictions
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
    remove_unused_columns=False,  # Prevent removing unused columns
)                                                                                                                                                                                                                                              ;from transformers import MarianMTModel ;model=MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# Initialize the model
vocab_size = tokenizer.vocab_size
model = TranslationModel(vocab_size, hidden_size, num_layers, num_heads)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=(len(train_data) // batch_size) * num_epochs
)

# Initialize the BLEU metric
metric = evaluate.load("sacrebleu")
# Track BLEU score over epochs
bleu_scores = []

# Define the Trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,  # Add the evaluation dataset
    tokenizer=tokenizer,
)

# Evaluate and track BLEU after each epoch
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} - Training...")
    trainer.train()
    
    # After each epoch, evaluate and compute BLEU score
    print(f"Evaluating after Epoch {epoch + 1}...")
    predictions, label_ids, _ = trainer.predict(eval_data)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute BLEU score
    bleu_score = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    bleu_scores.append(bleu_score['score'])
    print(f"BLEU score after Epoch {epoch + 1}: {bleu_score['score']:.2f}")

# Plot BLEU score over epochs
plt.plot(range(1, num_epochs + 1), bleu_scores, label="BLEU Score", marker='o')
plt.xlabel("Epochs")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Epochs")
plt.legend()
plt.show()

# Save the model and tokenizer
model_dir = "./ar_en_model"
config = MarianConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=hidden_size,  # Hidden size of the model
    encoder_layers=num_layers,
    decoder_layers=num_layers,
    encoder_attention_heads=num_heads,
    decoder_attention_heads=num_heads,
    max_position_embeddings=max_seq_length,
)
config.save_pretrained(model_dir)
trainer.save_model(model_dir)  # Saves the model weights
tokenizer.save_pretrained(model_dir)  # Saves the tokenizer

print("Model, tokenizer, and training arguments saved using Trainer.")
 # Evaluate the model at the end of each epoch
train_losses = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} - Training...")
    trainer.train()
    
    # After each epoch, evaluate and get the loss
    print(f"Evaluating after Epoch {epoch + 1}...")
    eval_results = trainer.evaluate(eval_data)
    
    # Append the eval loss to the train_losses list
    train_losses.append(eval_results['eval_loss'])
    
    # Optionally, print the evaluation loss
    print(f"Evaluation Loss after Epoch {epoch + 1}: {eval_results['eval_loss']:.4f}")

# Plot learning curve (Loss vs. Epochs)
plt.plot(range(1, num_epochs + 1), train_losses, label="Eval Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve (Loss vs Epochs)")
plt.legend()
plt.show()
