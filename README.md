# Local Meeting Transcription Tool

A privacy-focused Streamlit application that transcribes meeting recordings using OpenAI's Whisper model running locally on your machine.

## Features

- Upload audio recordings (MP3, WAV, M4A, OGG)
- Transcribe using various Whisper model sizes (tiny, base, small, medium, large)
- View and download complete transcripts
- View and download transcripts with timestamps
- GPU acceleration (if available)
- **100% Local Processing** - Your data never leaves your computer

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: To use GPU acceleration, ensure you have the appropriate CUDA toolkit installed.

## Usage

1. Run the Streamlit application:
```bash
streamlit run local_transcript_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Follow the on-screen instructions to upload and transcribe your audio files

## Model Selection

- **Tiny**: Fastest but least accurate (~39MB download)
- **Base**: Good balance of speed and accuracy for most casual use cases (~142MB download)
- **Small**: Better accuracy, moderately slower (~466MB download)
- **Medium**: High accuracy, significantly slower (~1.5GB download)
- **Large**: Highest accuracy, slowest and most resource-intensive (~3GB download)

## Privacy & Security

Unlike cloud-based transcription services:

- All processing happens locally on your computer
- Your audio files never leave your machine
- No API keys or account registration required
- Works offline after initial model download
- Compliant with strict security requirements

## Dependencies

The application requires:
- Python 3.8+
- FFmpeg (for audio processing)
- PyTorch
- Whisper

## Notes

- Large audio files and larger models will require more processing time and memory
- GPU acceleration will significantly improve processing speed
- First-time use will download the selected Whisper model, which may take some time

## Credits

This application uses OpenAI's Whisper model for speech recognition and Streamlit for the web interface. # nola
