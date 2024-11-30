import whisper
import pyaudio
from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
from google.cloud import texttospeech
import torch

# Realtime STT with Whisper 
def real_time_transcribe_whisper(model_name="small"):
    model = whisper.load_model(model_name)

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
    print(f"Transcribed Text: {transcribed_text}")

    return transcribed_text

# MarianMT translation 
class Translator:
    def __init__(self):
        # self.mul_to_mul_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        # self.mul_to_mul_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        # self.en_to_mul_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        # self.en_to_mul_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        # self.mul_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        # self.mul_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        # # will be deleted ⬇️ 
        self.en_to_ar_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.en_to_ar_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.ar_to_en_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.ar_to_en_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        
        #Wiil be changed ⬇️
        
    def translate(self, text, source_language="en", target_language="ar"):
        if target_language == "ar":
            inputs = self.en_to_ar_tokenizer(text, return_tensors="pt")
            outputs = self.en_to_ar_model.generate(**inputs)
            return self.en_to_ar_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = self.ar_to_en_tokenizer(text, return_tensors="pt")
            outputs = self.ar_to_en_model.generate(**inputs)
            return self.ar_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

# TTS with Google Cloud WaveNet
def synthesize_speech_wavenet(text, output_file="output.mp3", language_code="en-US", voice_name="en-US-Wavenet-D"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name, ssml_gender=texttospeech.SsmlVoiceGender.MALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to '{output_file}'")


# functions combination 
def real_time_translate_and_respond(target_language, output_audio_path="translated_output.wav"):
    # Real-time transcription with Whisper
    transcribed_text = real_time_transcribe_whisper()
    print(f"Transcribed Text: {transcribed_text}")

    # Translation
    translator = Translator()
    translated_text = translator.translate(transcribed_text, target_language=target_language)
    print(f"Translated Text: {translated_text}")

    
    # Text-to-Speech synthesis
    language_code = "ar-XA" if target_language == "ar" else "en-US"
    voice_name = "ar-XA-Wavenet-A" if target_language == "ar" else "en-US-Wavenet-D"
    synthesize_speech_wavenet(translated_text, output_audio_path, language_code=language_code, voice_name=voice_name)


# Example usage
real_time_translate_and_respond(target_language="en")
