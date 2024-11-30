from transformers import MarianMTModel, MarianTokenizer
import torch

# Step 1: Load the saved model and tokenizer
model_dir = "./fine_tuned_ar_en_model"  # Path to your saved model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir)

# Step 2: Define a function for translation
def translate(text, model, tokenizer):
    # Tokenize the input text (English)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Generate the translation (Arabic)
    with torch.no_grad():  # No need to calculate gradients during inference
        translated = model.generate(**inputs)
    
    # Decode the translated tokens to get the Arabic text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Step 3: Translate new English text
source_text = "امرأة ترتدي شبكة على رأسها تقطع كعكة"   # Example text
translated_text = translate(source_text, model, tokenizer)

# Print the translated text
print(f"Original (Ar): {source_text}")
print(f"Translated (En): {translated_text}")
