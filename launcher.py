import streamlit as st
import subprocess
import sys
import os

st.set_page_config(
    page_title="Meeting Transcriber Launcher",
    page_icon="🎙️",
    layout="centered",
)

st.title("Meeting Transcription Tool")
st.markdown("Choose which version of the transcription tool to launch.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Local Whisper Model")
    st.markdown("""
    - Runs entirely on your computer
    - No API key required
    - No usage costs
    - Requires more computing resources
    - Multiple model sizes available
    """)
    if st.button("Launch Local Version", use_container_width=True):
        st.success("Launching local Whisper model version...")
        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to transcript_app.py
        app_path = os.path.join(script_dir, "transcript_app.py")
        # Launch the app
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        st.markdown("If a new browser tab doesn't open automatically, the app should be running at http://localhost:8501")

with col2:
    st.markdown("### OpenAI API Version")
    st.markdown("""
    - Uses OpenAI's cloud-based models
    - Requires an OpenAI API key
    - Incurs usage costs
    - Faster processing
    - No local resources needed
    """)
    if st.button("Launch OpenAI API Version", use_container_width=True):
        st.success("Launching OpenAI API version...")
        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to openai_transcript_app.py
        app_path = os.path.join(script_dir, "openai_transcript_app.py")
        # Launch the app
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        st.markdown("If a new browser tab doesn't open automatically, the app should be running at http://localhost:8501")

st.markdown("---")
st.markdown("### Installation")
with st.expander("Installation and Setup Instructions"):
    st.markdown("""
    If you haven't already installed the required dependencies, run:
    ```bash
    pip install -r requirements.txt
    ```
    
    For the local version to use GPU acceleration, ensure you have:
    1. A compatible NVIDIA GPU
    2. The appropriate CUDA toolkit installed
    3. PyTorch with CUDA support
    
    For the OpenAI API version, you'll need:
    - An OpenAI account with API access
    - A valid API key
    """)

st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's Whisper") 