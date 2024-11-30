from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Step 1: Load the MS-Glossary-EN-AR dataset
dataset = load_dataset("ymoslem/MS-Glossary-EN-AR", split="train[:100%]")  # Use 1% for faster training
dataset = dataset.shuffle(seed=42)

# Step 2: Load MarianMTModel and MarianTokenizer for English-to-Arabic translation
model_name = "Helsinki-NLP/opus-mt-en-ar"  # Use the EN-AR model

# Load tokenizer and model using MarianMT (specifically for EN-AR)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Step 3: Preprocessing function to tokenize the dataset
def preprocess_function(examples):
    # Tokenize the English and Arabic text
    inputs = tokenizer(examples['text_en'], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples['text_ar'], truncation=True, padding="max_length", max_length=128)

    # Set targets as labels for the model
    inputs['labels'] = targets['input_ids']
    return inputs

# Step 4: Prepare the training dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set dataset format for PyTorch
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 5: Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save the model
    evaluation_strategy="no",  # Disable evaluation during training
    save_strategy="steps",  # Save model at regular intervals
    per_device_train_batch_size=8,  # Smaller batch size to fit in memory
    per_device_eval_batch_size=8,  # Evaluation batch size
    num_train_epochs=3,  # Number of training epochs
    save_total_limit=2,  # Keep only the last 2 saved models
    fp16=True,  # Mixed precision training for faster performance
    predict_with_generate=True,  # Use generation for predictions
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
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
model.save_pretrained("./fine_tuned_en_ar_model")
tokenizer.save_pretrained("./fine_tuned_en_ar_model")

print("Training complete and model saved!")

print("Testing the model...")
text = "i love you very much"  # Test sentence in Arabic
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)
print("Translation:", tokenizer.decode(outputs[0], skip_special_tokens=True))