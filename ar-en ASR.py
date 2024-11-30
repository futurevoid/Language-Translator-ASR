from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Step 1: Load Arabic-to-English dataset
dataset = load_dataset("opus100", "ar-en", split="train[:11%]")  # Use 50% of the data for training
dataset = dataset.shuffle(seed=42)

# Step 2: Load MarianMTModel and MarianTokenizer for Arabic-to-English translation
model_name = "Helsinki-NLP/opus-mt-ar-en"

# Load tokenizer and model using MarianMT (specifically for ar-en)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Step 3: Preprocessing function to tokenize the dataset
def preprocess_function(examples):
    # Tokenize the Arabic text and English text in batches
    inputs = tokenizer(examples['ar'], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples['en'], truncation=True, padding="max_length", max_length=128)

    # Set targets as labels for the model
    inputs['labels'] = targets['input_ids']
    return inputs

# Step 4: Prepare the training dataset
tokenized_dataset = dataset.map(lambda examples: preprocess_function({
    'ar': [item['ar'] for item in examples['translation']],
    'en': [item['en'] for item in examples['translation']]
}), batched=True)

# Set dataset format
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 5: Define the training arguments with no evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model
    evaluation_strategy="no",  # Disable evaluation
    save_strategy="steps",  # Save model every few steps
    per_device_train_batch_size=8,  # Use smaller batch size for limited GPU
    per_device_eval_batch_size=8,  # Evaluation batch size
    num_train_epochs=3,  # Number of epochs
    save_total_limit=2,  # Keep only the last 2 saved models
    fp16=True,  # Use mixed precision training for faster training
    predict_with_generate=True,  # Use generate method for predictions
    logging_dir="./logs",  # Log directory
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps to simulate larger batch size
    dataloader_num_workers=4,  # Increase the number of workers to load data in parallel
)

# Step 6: Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Step 7: Start training the model
trainer.train()

# Step 8: Save the trained model and tokenizer
model.save_pretrained("./fine_tuned_ar_en_model")
tokenizer.save_pretrained("./fine_tuned_ar_en_model")

print("Training complete and model saved!")
