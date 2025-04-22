import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment
import numpy as np
import torch

st.set_page_config(
    page_title="Meeting Transcriber",
    page_icon="🎙️",
    layout="wide",
)

# Title and description
st.title("Meeting Transcription Tool")
st.markdown("Upload an audio recording of your meeting and get a transcript.")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Using device: {device}")

# Model selection
model_size = st.selectbox(
    "Select Whisper model size:",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
    help="Larger models are more accurate but slower and require more memory"
)

@st.cache_resource
def load_model(model_size):
    return whisper.load_model(model_size, device=device)

# Load the model with caching
with st.spinner(f"Loading Whisper {model_size} model..."):
    model = load_model(model_size)

# File uploader
uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing... This may take a while depending on the file size and model."):
            # Transcribe the audio
            result = model.transcribe(audio_path)
            transcript = result["text"]
        
        # Display transcript
        st.subheader("Transcript")
        st.text_area("", transcript, height=300)
        
        # Download button for transcript
        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        # Display segments with timestamps
        st.subheader("Segments with Timestamps")
        segments_data = ""
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            segments_data += f"[{start:.2f}s - {end:.2f}s] {text}\n"
        
        st.text_area("", segments_data, height=300)
        
        # Download button for segments
        st.download_button(
            label="Download Segments with Timestamps",
            data=segments_data,
            file_name="transcript_with_timestamps.txt",
            mime="text/plain"
        )
        
        # Clean up the temporary file
        os.unlink(audio_path)

# Add a section with instructions
with st.expander("Instructions"):
    st.markdown("""
    ## How to use this app
    1. Select a Whisper model size (larger models are more accurate but slower)
    2. Upload an audio file of your meeting
    3. Click the 'Transcribe Audio' button
    4. View and download the transcript
    
    ## Supported File Formats
    - MP3
    - WAV
    - M4A
    - OGG
    
    ## About Whisper
    Whisper is an automatic speech recognition (ASR) system trained on a large dataset of diverse audio. It's designed to transcribe speech in multiple languages and can handle various accents, background noise, and technical language.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's Whisper") 