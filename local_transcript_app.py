import streamlit as st
import whisper
import tempfile
import os
import torch
import numpy as np
import subprocess
import wave
import uuid

st.set_page_config(
    page_title="Local Meeting Transcriber",
    page_icon="🎙️",
    layout="wide",
)

# Title and description
st.title("Local Meeting Transcription Tool")
st.markdown("Upload an audio recording of your meeting and get a transcript. All processing happens locally on your machine.")

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

# Convert m4a to wav using ffmpeg
def convert_m4a_to_wav(input_path):
    # Create unique output path to prevent collisions
    unique_id = str(uuid.uuid4())
    output_path = f"/tmp/converted_{unique_id}.wav"
    
    try:
        # Use ffmpeg to convert m4a to wav
        cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path]
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting M4A to WAV: {e.stderr.decode()}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "m4a"])

if uploaded_file is not None:
    # Determine file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file, format=f"audio/{file_extension}")
    
    if st.button("Transcribe Audio"):
        try:
            # For M4A files, convert to WAV first
            if file_extension == "m4a":
                with st.spinner("Converting M4A to WAV format..."):
                    wav_path = convert_m4a_to_wav(audio_path)
                    if wav_path is None:
                        st.error("Failed to convert M4A to WAV. Please try a different file.")
                        os.unlink(audio_path)
                        st.stop()
                    # Use the converted WAV for transcription
                    transcription_path = wav_path
            else:
                # Use original file for WAV
                transcription_path = audio_path
            
            # Helper function to read WAV file directly using wave library
            def load_wav_file(file_path):
                with wave.open(file_path, 'rb') as wav_file:
                    # Get basic information
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    # Read all frames
                    frames = wav_file.readframes(n_frames)
                    
                    # Convert bytes to numpy array
                    if sample_width == 1:  # 8-bit unsigned
                        dtype = np.uint8
                        data = np.frombuffer(frames, dtype=dtype)
                        data = (data.astype(np.float32) - 128) / 128.0  # Convert to float in [-1, 1]
                    elif sample_width == 2:  # 16-bit signed
                        dtype = np.int16
                        data = np.frombuffer(frames, dtype=dtype)
                        data = data.astype(np.float32) / 32768.0  # Convert to float in [-1, 1]
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # If stereo, convert to mono by averaging channels
                    if n_channels == 2:
                        data = data.reshape(-1, 2).mean(axis=1)
                    
                    # Resample to 16kHz if different
                    if frame_rate != 16000:
                        # Simple resampling by linear interpolation
                        original_length = len(data)
                        target_length = int(original_length * 16000 / frame_rate)
                        indices = np.linspace(0, original_length - 1, target_length)
                        data = np.interp(indices, np.arange(original_length), data)
                    
                    return data

            with st.spinner("Processing audio..."):
                # Try to load the audio using Whisper's default method first
                try:
                    with st.spinner("Transcribing... This may take a while depending on the file size and model."):
                        result = model.transcribe(transcription_path)
                        transcript = result["text"]
                except Exception as whisper_error:
                    st.warning(f"Default audio loading failed: {str(whisper_error)}. Trying alternative method...")
                    
                    # Fallback to our custom WAV loader
                    try:
                        audio_data = load_wav_file(transcription_path)
                        with st.spinner("Transcribing with alternative method..."):
                            result = model.transcribe(audio_data)
                            transcript = result["text"]
                    except Exception as e:
                        st.error(f"Alternative method also failed: {str(e)}")
                        raise
                
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
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up the temporary files
            os.unlink(audio_path)
            # Clean up converted file if it exists
            if file_extension == "m4a" and 'wav_path' in locals() and wav_path is not None:
                try:
                    os.unlink(wav_path)
                except:
                    pass

# Add a section with instructions
with st.expander("Instructions"):
    st.markdown("""
    ## How to use this app
    1. Select a Whisper model size (larger models are more accurate but slower)
    2. Upload a WAV or M4A audio file (such as Voice Memos from macOS)
    3. Click the 'Transcribe Audio' button
    4. View and download the transcript
    
    ## Supported File Formats
    - WAV files
    - M4A files (macOS Voice Memos)
    
    ## About Whisper
    Whisper is an automatic speech recognition (ASR) system trained on a large dataset of diverse audio. It's designed to transcribe speech in multiple languages and can handle various accents, background noise, and technical language.
    
    ## Privacy and Security
    All processing is done locally on your machine. Your audio files and transcriptions never leave your computer, ensuring complete privacy and security.
    """)

# Add a section about security
with st.expander("Privacy & Security"):
    st.markdown("""
    ## Local Processing
    This application uses OpenAI's Whisper model that runs entirely on your local machine. None of your audio data is sent to external servers or APIs.
    
    ## Why Local Processing Matters
    - **Data Privacy**: Your sensitive meeting recordings never leave your computer
    - **Security**: No risk of data interception during transfer
    - **Compliance**: Helps meet regulatory requirements for sensitive data
    - **No Internet Required**: Works offline once the model is downloaded
    
    ## Technical Details
    The Whisper model is downloaded once and stored locally. All audio processing, transcription, and text generation happens on your CPU or GPU, depending on availability.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's Whisper (running locally)") 