import streamlit as st
import tempfile
import os
import openai
import time

st.set_page_config(
    page_title="Meeting Transcriber (OpenAI API)",
    page_icon="🎙️",
    layout="wide",
)

# Title and description
st.title("Meeting Transcription Tool (OpenAI API)")
st.markdown("Upload an audio recording of your meeting and get a transcript using OpenAI's API.")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    openai.api_key = api_key

# File uploader
uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None and api_key:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # Model selection for OpenAI
    model = st.selectbox(
        "Select Whisper model:",
        ["whisper-1"],
        index=0
    )
    
    if st.button("Transcribe Audio"):
        try:
            with st.spinner("Transcribing... This may take a while depending on the file size."):
                # Open the file
                with open(audio_path, "rb") as audio_file:
                    # Transcribe the audio using OpenAI API
                    response = openai.Audio.transcribe(
                        model=model,
                        file=audio_file,
                        response_format="verbose_json"
                    )
                
                # Extract transcript
                transcript = response["text"]
                
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
            if "segments" in response:
                st.subheader("Segments with Timestamps")
                segments_data = ""
                for segment in response["segments"]:
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
            # Clean up the temporary file
            os.unlink(audio_path)
elif uploaded_file is not None and not api_key:
    st.warning("Please enter your OpenAI API key to transcribe the audio.")

# Add a section with instructions
with st.expander("Instructions"):
    st.markdown("""
    ## How to use this app
    1. Enter your OpenAI API key
    2. Upload an audio file of your meeting
    3. Click the 'Transcribe Audio' button
    4. View and download the transcript
    
    ## Supported File Formats
    - MP3
    - WAV
    - M4A
    - OGG
    
    ## About OpenAI's Whisper API
    The OpenAI Whisper API provides state-of-the-art speech-to-text capabilities. Using the API:
    - Processing is done in OpenAI's cloud (no local resources needed)
    - Requires an OpenAI API key and will incur usage costs
    - Typically provides faster transcription than local processing
    
    ## API Key Security
    Your API key is not stored and is only used for the current session. Always keep your API keys secure.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's Whisper API") 