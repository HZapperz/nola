#!/bin/bash

# Exit script if any command fails
set -e

echo "Setting up FFmpeg and dependencies for the Meeting Transcription Tool on macOS"
echo "==============================================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew already installed."
fi

# Install FFmpeg using Homebrew
echo "Installing FFmpeg..."
brew install ffmpeg

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "==============================================================================="
echo "Setup complete! You can now run the application with: streamlit run local_transcript_app.py"
echo "-------------------------------------------------------------------------------"
echo "Note: You'll need to run 'source venv/bin/activate' in any new terminal sessions"
echo "before running the application."
echo "===============================================================================" 