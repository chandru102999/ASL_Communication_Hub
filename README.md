🤟 ASL Communication Hub

The ASL Communication Hub is a comprehensive, real-time application built with Streamlit and computer vision libraries (OpenCV, MediaPipe) to bridge communication between sign language and spoken language. It offers two primary modes: Sign to Text and Audio to Sign.

✨ Features

1. 👐 Sign to Text Mode (ASL Recognition)

Real-time Hand Tracking: Uses MediaPipe to detect and track hand landmarks via the webcam.

ASL Alphabet Recognition: Employs a trained Machine Learning model (model.pkl) to predict the ASL letter (A-Z, Space, Period) being signed.

Interactive Input: Users build words and sentences by pressing keyboard keys (Space, Enter, Q) while signing.

Text-to-Speech (TTS): Converts the final detected sentence into spoken audio using gTTS.

CV Feedback: Provides visual cues, instruction overlays, and a guidance box on the live camera feed.

2. 🗣️ Audio to Sign Mode (Spoken Word to ASL)

Voice Transcription: Uses the device microphone and the Google Speech Recognition API for direct voice input.

File Transcription: Supports uploading audio files (mp3, wav, ogg) and transcribes them using the powerful Whisper model (via the Python package).

Visual Output: Converts the transcribed text into:

A sequence of static ASL Alphabet Images for letter-by-letter interpretation.

Matching dynamic ASL Word/Phrase GIFs if available in the resource folder.

🚀 Setup and Installation

This project requires not only the Python dependencies but also specific resource files (.pkl model, images, and GIFs) to function correctly.

Prerequisites

Python 3.8+

A working webcam for the Sign to Text mode.

Resource File Structure (CRITICAL)

You must create the following file structure in your project's root directory:

.
├── app.py          # The main Streamlit code
├── model.p         # REQUIRED: Trained ML model file (the code expects this)
├── images/         # REQUIRED: Folder containing ASL alphabet images (e.g., A.png, B.png)
└── gifs/           # REQUIRED: Folder containing ASL word/phrase GIFs (e.g., HELLO.gif, THANKYOU.gif)


Note: The application will fail with a FileNotFoundError if the model.p file or the images/ and gifs/ directories are missing or incorrectly named.

Installation Steps

Install Dependencies:

The application uses several heavy libraries, including mediapipe, opencv-python, and whisper.

pip install streamlit opencv-python mediapipe numpy pickle gtts whisper-cpp-python Pillow speechrecognition sounddevice
# Note: Depending on your OS, installing mediapipe and sounddevice might require system dependencies.


(If running the code in a sandbox environment, you might need to adjust dependencies based on available pre-installed packages.)

Download or Train the Model:
Ensure you have your trained hand classification model saved as model.p in the root directory.

⚙️ Usage

Run the Streamlit App:

streamlit run app.py


Select Mode: Use the sidebar to switch between "Sign to Text" and "Audio to Sign" modes.

👐 Sign to Text Instructions

Click "✨ Start Sign Detection".

The camera window (an external window powered by OpenCV) will open.

Place your hand inside the large green box.

To input a detected letter, press the SPACEBAR.

To finish a word and add a space to the sentence buffer, press ENTER.

To finish the entire sentence and close the camera, press Q.

The transcribed text and the generated speech audio will appear in the Streamlit interface.

🗣️ Audio to Sign Instructions

Select the "Audio to Sign" mode.

Choose the Voice Input tab to use your microphone (requires browser permission) or the File Upload tab to upload an audio file.

The application will transcribe the speech and display the resulting ASL image sequence and any matching word GIFs.
