from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

# Load the MS-Glossary-EN-AR dataset
dataset = load_dataset("ymoslem/MS-Glossary-EN-AR")

# Split the dataset into train (80%) and validation (20%)
train_dataset = dataset["train"].select(range(int(0.8 * len(dataset["train"])))) 
val_dataset = dataset["train"].select(range(int(0.8 * len(dataset["train"]))), len(dataset["train"]))

# Load MarianMTModel and MarianTokenizer for English-to-Arabic translation
model_name = "Helsinki-NLP/opus-mt-en-ar"  # Use the EN-AR model

# Load tokenizer and model using MarianMT (specifically for EN-AR)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocessing function to tokenize the dataset
def preprocess_function(examples):
    # Tokenize the English and Arabic text
    inputs = tokenizer(examples['text_en'], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples['text_ar'], truncation=True, padding="max_length", max_length=128)

    # Set targets as labels for the model
    inputs['labels'] = targets['input_ids']
    return inputs

# Prepare the tokenized dataset
train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)

# Set dataset format for PyTorch
train_tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="steps",  # Save model at regular intervals
    per_device_train_batch_size=8,  # Smaller batch size to fit in memory
    per_device_eval_batch_size=8,  # Evaluation batch size
    num_train_epochs=1,  # Number of training epochs
    save_total_limit=2,  # Keep only the last 2 saved models
    fp16=True,  # Mixed precision training for faster performance
    predict_with_generate=True,  # Use generation for predictions
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./fine_tuned_en_ar_model")
tokenizer.save_pretrained("./fine_tuned_en_ar_model")

print("Training complete and model saved!")

# Evaluating the model
print("Evaluating the model...")
metric = evaluate.load("sacrebleu")  # Load BLEU metric using evaluate

# Prepare a small evaluation dataset (use validation set)
eval_dataset = val_tokenized_dataset.select(range(100))  # Use the first 100 samples for evaluation
predictions, label_ids, metrics = trainer.predict(eval_dataset)

# Decode predictions and labels
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

# Compute BLEU score
bleu_score = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
print(f"BLEU score: {bleu_score['score']:.2f}")
