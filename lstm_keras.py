from datasets import load_dataset
from keras.models import Model  # type: ignore
from keras.layers import Input, LSTM, Dense  # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json

batch_size = 64  # Batch size for training.
epochs = 150  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

# Load the dataset
dataset = load_dataset("ymoslem/MS-Glossary-EN-AR", split="train")

# Define the train-test split (80:20)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))  # First 80% of the data
val_dataset = dataset.select(range(train_size, len(dataset)))  # Last 20% of the data

# Check the sizes of train and validation datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Preprocess the training dataset
train_input_texts = []
train_target_texts = []

for i in range(len(train_dataset)):
    input_text = train_dataset[i]['text_en']
    target_text = train_dataset[i]['text_ar']
    target_text = "\t" + target_text + "\n"  # Add start and end tokens
    train_input_texts.append(input_text)
    train_target_texts.append(target_text)

# Preprocess the validation dataset
val_input_texts = []
val_target_texts = []

for i in range(len(val_dataset)):
    input_text = val_dataset[i]['text_en']
    target_text = val_dataset[i]['text_ar']
    target_text = "\t" + target_text + "\n"  # Add start and end tokens
    val_input_texts.append(input_text)
    val_target_texts.append(target_text)

# Tokenize characters for both train and validation datasets
input_characters = set()
target_characters = set()

for input_text in train_input_texts + val_input_texts:
    for char in input_text:
        input_characters.add(char)
for target_text in train_target_texts + val_target_texts:
    for char in target_text:
        target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in train_input_texts + val_input_texts])
max_decoder_seq_length = max([len(txt) for txt in train_target_texts + val_target_texts])

print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# One-hot encode the training data
encoder_input_train = np.zeros(
    (len(train_input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_train = np.zeros(
    (len(train_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_train = np.zeros(
    (len(train_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(train_input_texts, train_target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_train[i, t, input_token_index[char]] = 1.
    encoder_input_train[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        decoder_input_train[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_train[i, t - 1, target_token_index[char]] = 1.
    decoder_input_train[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_train[i, t:, target_token_index[' ']] = 1.

# One-hot encode the validation data
encoder_input_val = np.zeros(
    (len(val_input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_val = np.zeros(
    (len(val_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_val = np.zeros(
    (len(val_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_val[i, t, input_token_index[char]] = 1.
    encoder_input_val[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        decoder_input_val[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_val[i, t - 1, target_token_index[char]] = 1.
    decoder_input_val[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_val[i, t:, target_token_index[' ']] = 1.

# Define the model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
rdlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
history = model.fit([encoder_input_train, decoder_input_train], decoder_target_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([encoder_input_val, decoder_input_val], decoder_target_val),
                    callbacks=[early_stopping,
                               checkpoint, 
                               rdlr])

# Save token indices
with open('input_token_index.json', 'w') as f:
    json.dump(input_token_index, f)
with open('target_token_index.json', 'w') as f:
    json.dump(target_token_index, f)
# Load the best model
model.load_weights('best_model.keras',)
    
# Get predictions on the validation set
out_predict = model.predict([encoder_input_val, decoder_input_val])

# Convert the predictions from one-hot encoding to indices
out_predict_indices = np.argmax(out_predict, axis=-1)

# Convert the ground truth data from one-hot encoding to indices
decoder_target_val_indices = np.argmax(decoder_target_val, axis=-1)

# Flatten both predictions and true labels for comparison
out_predict_flat = out_predict_indices.flatten()
decoder_target_flat = decoder_target_val_indices.flatten()

# Calculate accuracy
score = accuracy_score(decoder_target_flat, out_predict_flat)
print(f'Test Accuracy: {round(score, 4)}')

# Plotting Learning Curves for Loss and Accuracy
def plot_learning_curve(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_learning_curve(history)


