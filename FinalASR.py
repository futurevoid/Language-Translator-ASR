import whisper
import pyaudio
from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
import soundfile as sf
from google.cloud import texttospeech
import torch

# --- Real-Time Speech-to-Text (STT) with Whisper ---
def real_time_transcribe_whisper(model_name="turbo"):
    """Capture audio from microphone and transcribe in real-time using Whisper."""
    model = whisper.load_model(model_name)

    # Set up PyAudio to capture audio in real-time
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    stream.start_stream()

    print("Listening... (press Ctrl+C to stop)")

    frames = []
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            frames.append(np.frombuffer(data, np.int16))

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording stopped.")

    # Process the recorded audio using Whisper
    audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0
    result = model.transcribe(audio_data, fp16=False)
    transcribed_text = result['text']
    return transcribed_text


def translate(text, model, tokenizer):
    
    # Tokenize the input text (English)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Generate the translation (Arabic)
    with torch.no_grad():  # No need to calculate gradients during inference
        translated = model.generate(**inputs)
    
    # Decode the translated tokens to get the Arabic text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


# --- Text-to-Speech (TTS) with Google Cloud WaveNet ---
def synthesize_speech_wavenet(text, output_file="output.mp3", language_code="en-US", voice_name="en-US-Wavenet-D"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name, ssml_gender=texttospeech.SsmlVoiceGender.MALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to '{output_file}'")


# --- Combined Process for Real-Time Translation and Speech ---
def real_time_translate_and_respond(target_language="en", output_audio_path="translated_output.wav"):
    # Real-time transcription with Whisper
    transcribed_text = real_time_transcribe_whisper()
    print(f"Transcribed Text: {transcribed_text}")

    # Translation
    if target_language == "en":
        model_dir = "./fine_tuned_ar_en_model"
    else:
        model_dir = "./fine_tuned_en_ar_model"
        
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    
    translated_text = translate(transcribed_text, model, tokenizer)
    print(f"Translated Text: {translated_text}")

    
    # Text-to-Speech synthesis
    language_code = "ar-XA" if target_language == "ar" else "en-US"
    voice_name = "ar-XA-Wavenet-A" if target_language == "ar" else "en-US-Wavenet-D"
    synthesize_speech_wavenet(translated_text, output_audio_path, language_code=language_code, voice_name=voice_name)


# Example usage
# Set 'target_language' to "ar" for English-to-Arabic, and "en" for Arabic-to-English
real_time_translate_and_respond(target_language="ar")
