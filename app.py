import streamlit as st
import sounddevice as sd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load pre-trained model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Set up the microphone
duration = 20  # Recording duration in seconds
sample_rate = 16000  # Sample rate of the microphone

# Function to transcribe the recorded audio
def transcribe_audio(samples):
    input_values = tokenizer(samples, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

# Streamlit app
st.title("Real-time Voice Transcription")

# Start recording button
if st.button("Start Recording"):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.success("Recording finished!")

    # Transcribe the recorded audio
    transcription = transcribe_audio(recording.flatten())
    st.subheader("Transcription")
    st.write(transcription)

# Instructions
st.write("Click the 'Start Recording' button to begin transcribing your voice in real-time.")
st.write("The recording will automatically stop after 20 seconds.")
