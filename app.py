# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import threading
# import os
# import time
# from datetime import datetime
# from gtts import gTTS
# import tempfile
# import whisper
# import difflib
# import speech_recognition as sr
# from PIL import Image
# import math

# # Initialize session state
# if 'detection_active' not in st.session_state:
#     st.session_state.detection_active = False

# class ASLDetector:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.hands = self.mp_hands.Hands(static_image_mode=False, 
#                                        min_detection_confidence=0.5, 
#                                        max_num_hands=1)
#         self.model = pickle.load(open('./model.p', 'rb'))['model']
#         self.labels_dict = {
#             0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
#             8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
#             15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
#             22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: '.'
#         }

#     def process_hand_landmarks(self, hand_landmarks):
#         data_aux = []
#         x_ = []
#         y_ = []
        
#         for landmark in hand_landmarks.landmark:
#             x_.append(landmark.x)
#             y_.append(landmark.y)
        
#         for landmark in hand_landmarks.landmark:
#             data_aux.append(landmark.x - min(x_))
#             data_aux.append(landmark.y - min(y_))
        
#         return np.array(data_aux[:42])

#     def add_instructions_overlay(self, frame):
#         """Add transparent instruction overlay to the frame"""
#         height, width = frame.shape[:2]
        
#         # Create semi-transparent overlay for instructions
#         overlay = frame.copy()
#         instruction_box = np.zeros((200, width, 3), dtype=np.uint8)
#         cv2.rectangle(instruction_box, (0, 0), (width, 200), (40, 40, 40), -1)
        
#         instructions = [
#             "📝 Instructions:",
#             "1. Place hand in the green box",
#             "2. Press SPACE to capture a letter",
#             "3. Press ENTER to complete a word",
#             "4. Press Q to finish the sentence"
#         ]
        
#         # Add instructions to overlay
#         for i, text in enumerate(instructions):
#             color = (0, 255, 0) if i == 0 else (255, 255, 255)
#             cv2.putText(instruction_box, text, (20, 40 + i * 35),
#                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
#         # Blend instruction box with frame
#         alpha = 0.8
#         frame[:200, :] = cv2.addWeighted(frame[:200, :], alpha, 
#                                         instruction_box, 1 - alpha, 0)
#         return frame

#     def process_camera(self):
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Cannot access camera. Please check your connection.")
#             return None

#         # Set HD resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         word_buffer = ""
#         sentence_buffer = ""
#         last_capture_time = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Flip frame horizontally for more intuitive interaction
#             frame = cv2.flip(frame, 1)
            
#             # Add guidance box
#             height, width = frame.shape[:2]
#             box_size = int(min(width, height) * 0.6)
#             center_x, center_y = width // 2, height // 2
            
#             # Add instructions and overlays
#             frame = self.add_instructions_overlay(frame)
            
#             # Add detection box
#             cv2.rectangle(frame, 
#                          (center_x - box_size//2, center_y - box_size//2),
#                          (center_x + box_size//2, center_y + box_size//2), 
#                          (0, 255, 0), 3)
            
#             # Process hand detection
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(frame_rgb)
#             detected_char = '?'
            
#             if results.multi_hand_landmarks:
#                 hand_landmarks = results.multi_hand_landmarks[0]
#                 self.mp_drawing.draw_landmarks(frame, hand_landmarks, 
#                                             self.mp_hands.HAND_CONNECTIONS)
                
#                 prediction = self.model.predict([self.process_hand_landmarks(hand_landmarks)])
#                 detected_char = self.labels_dict.get(int(prediction[0]), '?')
                
#                 # Show detection result
#                 cv2.putText(frame, f"Detected: {detected_char}", 
#                            (20, height - 60), cv2.FONT_HERSHEY_DUPLEX, 
#                            1, (0, 255, 0), 2)
            
#             # Show current word and sentence
#             cv2.putText(frame, f"Current Word: {word_buffer}", 
#                        (20, height - 120), cv2.FONT_HERSHEY_DUPLEX, 
#                        1, (255, 255, 0), 2)
            
#             # Show sentence with word wrap
#             sentence_y = height - 90
#             sentence_text = f"Sentence: {sentence_buffer}"
#             cv2.putText(frame, sentence_text, 
#                        (20, sentence_y), cv2.FONT_HERSHEY_DUPLEX, 
#                        1, (0, 255, 255), 2)
            
#             cv2.imshow("ASL Detection", frame)
#             key = cv2.waitKey(1)
            
#             if key == ord(' '):  # Space to capture letter
#                 if time.time() - last_capture_time > 0.5 and detected_char != '?':
#                     word_buffer += detected_char
#                     last_capture_time = time.time()
#                     # Add visual feedback
#                     flash = np.ones_like(frame) * 255
#                     cv2.imshow("ASL Detection", flash)
#                     cv2.waitKey(50)
                    
#             elif key == 13:  # Enter to complete word
#                 if word_buffer:
#                     sentence_buffer = sentence_buffer + " " + word_buffer if sentence_buffer else word_buffer
#                     word_buffer = ""
                    
#             elif key == ord('q'):  # Q to finish
#                 if word_buffer:
#                     sentence_buffer = sentence_buffer + " " + word_buffer if sentence_buffer else word_buffer
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         return sentence_buffer.strip()

# class AudioProcessor:
#     def __init__(self):
#         self.whisper_model = whisper.load_model("base")
        
#     def transcribe_audio(self, audio_data):
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
#                 temp_file.write(audio_data)
#                 temp_file.flush()
                
#                 result = self.whisper_model.transcribe(temp_file.name)
#                 os.unlink(temp_file.name)
#                 return result["text"]
#         except Exception as e:
#             st.error(f"Error transcribing audio: {str(e)}")
#             return None

# def load_resources():
#     """Load ASL images and GIFs"""
#     images = {}
#     gifs = {}
    
#     # Load ASL letter images
#     for img in os.listdir("images"):
#         if img.lower().endswith(('.png', '.jpg', '.jpeg')):
#             path = os.path.join("images", img)
#             label = os.path.splitext(img)[0].upper()
#             images[label] = path
    
#     # Load word/phrase GIFs
#     for gif in os.listdir("gifs"):
#         if gif.lower().endswith('.gif'):
#             path = os.path.join("gifs", gif)
#             label = os.path.splitext(gif)[0].upper()
#             gifs[label] = path
    
#     return images, gifs

# # def display_asl_sequence(text, images):
# #     """Display ASL letter sequence with improved layout"""
# #     if not text:
# #         return
    
# #     text = text.upper()
# #     words = text.split()
    
# #     for word_idx, word in enumerate(words):
# #         if word_idx > 0:
# #             st.markdown("<div style='margin: 20px 0;'></div>")
        
# #         st.markdown(f"""
# #             <div style="background-color: #f0f8ff; 
# #                        padding: 10px; 
# #                        border-radius: 10px; 
# #                        margin-bottom: 10px">
# #                 <h3 style="margin: 0; color: #2e7d32; text-align: center">
# #                     {word}
# #                 </h3>
# #             </div>
# #         """, unsafe_allow_html=True)
        
# #         max_chars_per_row = min(len(word), 6)
# #         num_rows = math.ceil(len(word) / max_chars_per_row)
        
# #         for row in range(num_rows):
# #             start_idx = row * max_chars_per_row
# #             end_idx = min((row + 1) * max_chars_per_row, len(word))
# #             word_segment = word[start_idx:end_idx]
            
# #             cols = st.columns(len(word_segment))
# #             for idx, char in enumerate(word_segment):
# #                 with cols[idx]:
# #                     if char.isalpha() and char in images:
# #                         img = Image.open(images[char])
# #                         img = img.resize((150, 150))
# #                         st.image(img, caption=char)
# #                     else:
# #                         st.write(char)

# def audio_to_sign():
#     """Handle audio input and conversion to ASL"""
#     images, gifs = load_resources()
#     audio_processor = AudioProcessor()
    
#     tab1, tab2 = st.tabs(["🎤 Voice Input", "📁 File Upload"])
    
#     with tab1:
#         st.subheader("Speak to See Signs")
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             if st.button("🎤 Start Recording", key="record"):
#                 with st.spinner("Listening..."):
#                     try:
#                         recognizer = sr.Recognizer()
#                         with sr.Microphone() as source:
#                             st.info("Speak now...")
#                             audio = recognizer.listen(source, timeout=5)
#                             text = recognizer.recognize_google(audio)
#                             show_sign_results(text, images, gifs)
#                     except Exception as e:
#                         st.error("Error recording audio. Please try again.")
    
#     with tab2:
#         st.subheader("Upload Audio File")
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             audio_file = st.file_uploader("Choose an audio file", 
#                                         type=["mp3", "wav", "ogg"])
#             if audio_file:
#                 st.audio(audio_file)
#                 with st.spinner("Processing audio..."):
#                     text = audio_processor.transcribe_audio(audio_file.read())
#                     if text:
#                         show_sign_results(text, images, gifs)

# def show_sign_results(text, images, gifs):
#     """Display ASL signs for detected text"""
#     if not text:
#         return
        
#     st.success(f"Detected text: {text}")
    
#     # Display ASL letter sequence
#     st.markdown("""
#         <div style="background-color: #e8f5e9; 
#                     padding: 20px; 
#                     border-radius: 10px; 
#                     margin: 20px 0;">
#             <h2 style="color: #2e7d32; margin: 0;">ASL Letter Sequence</h2>
#         </div>
#     """, unsafe_allow_html=True)
    
#     display_asl_sequence(text, images)
    
#     # Display matching GIF if available
#     st.markdown("""
#         <div style="background-color: #e8f5e9; 
#                     padding: 20px; 
#                     border-radius: 10px; 
#                     margin: 20px 0;">
#             <h2 style="color: #2e7d32; margin: 0;">Word/Phrase Sign</h2>
#         </div>
#     """, unsafe_allow_html=True)
    
#     text = text.upper()
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if text in gifs:
#             st.image(gifs[text], caption=text, use_container_width=True)
#         else:
#             matches = difflib.get_close_matches(text, list(gifs.keys()), 
#                                               n=1, cutoff=0.8)
#             if matches:
#                 st.image(gifs[matches[0]], caption=matches[0], 
#                         use_container_width=True)
#             else:
#                 st.info("No matching sign GIF found for this phrase")

# def main():
#     st.set_page_config(
#         page_title="ASL Communication Hub",
#         page_icon="🤟",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # Custom CSS for dark theme
#     st.markdown("""
#         <style>
#         /* Main theme colors */
#         :root {
#             --background: #1a1a1a;
#             --secondary-bg: #2d2d2d;
#             --accent: #7289da;
#             --text: #ffffff;
#             --text-secondary: #b0b0b0;
#             --success: #43b581;
#             --error: #f04747;
#             --warning: #faa61a;
#         }
        
#         /* Global styles */
#         .main {
#             background-color: var(--background);
#             color: var(--text);
#         }
        
#         .stButton > button {
#             background-color: var(--accent);
#             color: var(--text);
#             border-radius: 20px;
#             padding: 15px 32px;
#             font-weight: bold;
#             border: none;
#             transition: all 0.3s ease;
#         }
        
#         .stButton > button:hover {
#             background-color: #5b6eae;
#             transform: translateY(-2px);
#             box-shadow: 0 4px 8px rgba(114, 137, 218, 0.3);
#         }
        
#         /* Tab styling */
#         .stTabs [data-baseweb="tab-list"] {
#             gap: 24px;
#             background-color: var(--secondary-bg);
#             padding: 10px;
#             border-radius: 10px;
#         }
        
#         .stTabs [data-baseweb="tab"] {
#             padding: 10px 24px;
#             font-size: 18px;
#             font-weight: 500;
#             color: var(--text);
#         }
        
#         /* Headers */
#         h1, h2, h3, h4, h5, h6 {
#             color: var(--text);
#             font-weight: 600;
#         }
        
#         /* Info boxes */
#         .info-box {
#             background-color: var(--secondary-bg);
#             padding: 1.5rem;
#             border-radius: 10px;
#             margin: 1rem 0;
#             border-left: 5px solid var(--accent);
#             color: var(--text);
#         }
        
#         /* Success message */
#         .success-box {
#             background-color: rgba(67, 181, 129, 0.2);
#             padding: 1.5rem;
#             border-radius: 10px;
#             margin: 1rem 0;
#             border-left: 5px solid var(--success);
#             color: var(--text);
#         }
        
#         /* Word containers */
#         .word-container {
#             background-color: var(--secondary-bg);
#             padding: 15px;
#             border-radius: 10px;
#             margin: 10px 0;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
#         }
        
#         /* Sidebar */
#         .sidebar .sidebar-content {
#             background-color: var(--secondary-bg);
#         }
        
#         /* Custom components */
#         .custom-header {
#             background: linear-gradient(135deg, var(--accent), #5b6eae);
#             padding: 20px;
#             border-radius: 10px;
#             margin-bottom: 30px;
#             text-align: center;
#         }
        
#         .custom-header h1 {
#             color: white;
#             margin: 0;
#             font-size: 2.5em;
#             text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
#         }
        
#         .custom-subheader {
#             color: var(--text-secondary);
#             font-size: 1.2em;
#             margin-top: 10px;
#         }
        
#         /* File uploader */
#         .uploadedFile {
#             background-color: var(--secondary-bg);
#             border: 1px solid var(--accent);
#             border-radius: 10px;
#         }
        
#         /* Audio player */
#         audio {
#             width: 100%;
#             border-radius: 10px;
#             background-color: var(--secondary-bg);
#         }
        
#         /* Markdown text */
#         .stMarkdown {
#             color: var(--text);
#         }
        
#         /* Separator */
#         hr {
#             border-color: var(--secondary-bg);
#             margin: 20px 0;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # Sidebar with dark theme
#     st.sidebar.markdown("""
#         <div style="text-align: center; padding: 20px 0;">
#             <h1 style="color: #7289da;">🤟 ASL Hub</h1>
#         </div>
#     """, unsafe_allow_html=True)

#     mode = st.sidebar.radio(
#         "Select Communication Mode",
#         ["Sign to Text", "Audio to Sign"],
#         format_func=lambda x: "👐 " + x if x == "Sign to Text" else "🗣️ " + x,
#         key="mode_selection"
#     )

#     st.sidebar.markdown("""
#         <div style="margin-top: 30px; 
#                     padding: 20px; 
#                     background-color: rgba(114, 137, 218, 0.1); 
#                     border-radius: 10px; 
#                     border: 1px solid rgba(114, 137, 218, 0.2);">
#             <h4 style="color: #7289da; margin-top: 0;">About This App</h4>
#             <p style="color: #b0b0b0;">
#                 Bridge the communication gap between spoken language and 
#                 American Sign Language with real-time translation capabilities.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

#     # Main content with custom header
#     st.markdown("""
#         <div class="custom-header">
#             <h1>ASL Communication Hub</h1>
#             <p class="custom-subheader">
#                 Breaking down communication barriers with real-time ASL translation
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

#     if mode == "Sign to Text":
#         detector = ASLDetector()
        
#         st.markdown("""
#             <div class="info-box">
#                 <h3 style="margin-top: 0;">Sign Language Detection</h3>
#                 <p>Start the detection to convert your sign language gestures into text and speech.</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             if st.button("✨ Start Sign Detection", key="start_detection"):
#                 st.session_state.detection_active = True
#                 sentence = detector.process_camera()
#                 if sentence:
#                     st.markdown(f"""
#                         <div class="success-box">
#                             <h4 style="margin-top: 0;">Detected Text:</h4>
#                             <p style="font-size: 1.2em;">{sentence}</p>
#                         </div>
#                     """, unsafe_allow_html=True)
                    
#                     with st.spinner("Converting to speech..."):
#                         try:
#                             tts = gTTS(text=sentence, lang='en')
#                             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#                                 tts.save(fp.name)
#                                 st.audio(fp.name, format="audio/mp3")
#                                # os.remove(fp.name)
#                         except Exception as e:
#                             st.error(f"Error generating audio: {e}")
        
#         with col2:
#             st.markdown("""
#                 <div style="background-color: rgba(114, 137, 218, 0.1); 
#                            padding: 20px; 
#                            border-radius: 10px; 
#                            margin-top: 20px;
#                            border: 1px solid rgba(114, 137, 218, 0.2);">
#                     <h4 style="color: #7289da; margin-top: 0;">Quick Tips:</h4>
#                     <ul style="color: #b0b0b0;">
#                         <li>Keep your hand within the green box</li>
#                         <li>Ensure good lighting</li>
#                         <li>Make clear, deliberate signs</li>
#                         <li>Take your time between signs</li>
#                     </ul>
#                 </div>
#             """, unsafe_allow_html=True)
    
#     else:
#         audio_to_sign()

# def display_asl_sequence(text, images):
#     """Display ASL letter sequence with dark theme styling"""
#     if not text:
#         return
    
#     text = text.upper()
#     words = text.split()
    
#     for word_idx, word in enumerate(words):
#         if word_idx > 0:
#             st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
#         st.markdown(f"""
#             <div class="word-container">
#                 <h3 style="margin: 0; color: #7289da; text-align: center">
#                     {word}
#                 </h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         max_chars_per_row = min(len(word), 6)
#         num_rows = math.ceil(len(word) / max_chars_per_row)
        
#         for row in range(num_rows):
#             start_idx = row * max_chars_per_row
#             end_idx = min((row + 1) * max_chars_per_row, len(word))
#             word_segment = word[start_idx:end_idx]
            
#             cols = st.columns(len(word_segment))
#             for idx, char in enumerate(word_segment):
#                 with cols[idx]:
#                     if char.isalpha() and char in images:
#                         img = Image.open(images[char])
#                         img = img.resize((150, 150))
#                         st.image(img, caption=char,use_container_width=True)
#                     else:
#                         st.write(char)

# if __name__ == "__main__":
#     main()
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import os
import time
from datetime import datetime
from gtts import gTTS
import tempfile
import whisper
import difflib
import speech_recognition as sr
from PIL import Image
import math

# Initialize session state
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False

class ASLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                       min_detection_confidence=0.5, 
                                       max_num_hands=1)
        self.model = pickle.load(open('./model.p', 'rb'))['model']
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
            22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: '.'
        }

    def process_hand_landmarks(self, hand_landmarks):
        data_aux = []
        x_ = []
        y_ = []
        
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)
        
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))
        
        return np.array(data_aux[:42])

    def add_instructions_overlay(self, frame):
        """Add transparent instruction overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for instructions
        overlay = frame.copy()
        instruction_box = np.zeros((200, width, 3), dtype=np.uint8)
        cv2.rectangle(instruction_box, (0, 0), (width, 200), (40, 40, 40), -1)
        
        instructions = [
            "📝 Instructions:",
            "1. Place hand in the green box",
            "2. Press SPACE to capture a letter",
            "3. Press ENTER to complete a word",
            "4. Press Q to finish the sentence"
        ]
        
        # Add instructions to overlay
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(instruction_box, text, (20, 40 + i * 35),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Blend instruction box with frame
        alpha = 0.8
        frame[:200, :] = cv2.addWeighted(frame[:200, :], alpha, 
                                        instruction_box, 1 - alpha, 0)
        return frame

    def process_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your connection.")
            return None

        # Set HD resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        word_buffer = ""
        sentence_buffer = ""
        last_capture_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Add guidance box
            height, width = frame.shape[:2]
            box_size = int(min(width, height) * 0.6)
            center_x, center_y = width // 2, height // 2
            
            # Add instructions and overlays
            frame = self.add_instructions_overlay(frame)
            
            # Add detection box
            cv2.rectangle(frame, 
                         (center_x - box_size//2, center_y - box_size//2),
                         (center_x + box_size//2, center_y + box_size//2), 
                         (0, 255, 0), 3)
            
            # Process hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            detected_char = '?'
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                            self.mp_hands.HAND_CONNECTIONS)
                
                prediction = self.model.predict([self.process_hand_landmarks(hand_landmarks)])
                detected_char = self.labels_dict.get(int(prediction[0]), '?')
                
                # Show detection result
                cv2.putText(frame, f"Detected: {detected_char}", 
                           (20, height - 60), cv2.FONT_HERSHEY_DUPLEX, 
                           1, (0, 255, 0), 2)
            
            # Show current word and sentence
            cv2.putText(frame, f"Current Word: {word_buffer}", 
                       (20, height - 120), cv2.FONT_HERSHEY_DUPLEX, 
                       1, (255, 255, 0), 2)
            
            # Show sentence with word wrap
            sentence_y = height - 90
            sentence_text = f"Sentence: {sentence_buffer}"
            cv2.putText(frame, sentence_text, 
                       (20, sentence_y), cv2.FONT_HERSHEY_DUPLEX, 
                       1, (0, 255, 255), 2)
            
            cv2.imshow("ASL Detection", frame)
            key = cv2.waitKey(1)
            
            if key == ord(' '):  # Space to capture letter
                if time.time() - last_capture_time > 0.5 and detected_char != '?':
                    word_buffer += detected_char
                    last_capture_time = time.time()
                    # Add visual feedback
                    flash = np.ones_like(frame) * 255
                    cv2.imshow("ASL Detection", flash)
                    cv2.waitKey(50)
                    
            elif key == 13:  # Enter to complete word
                if word_buffer:
                    sentence_buffer = sentence_buffer + " " + word_buffer if sentence_buffer else word_buffer
                    word_buffer = ""
                    
            elif key == ord('q'):  # Q to finish
                if word_buffer:
                    sentence_buffer = sentence_buffer + " " + word_buffer if sentence_buffer else word_buffer
                break

        cap.release()
        cv2.destroyAllWindows()
        return sentence_buffer.strip()

class AudioProcessor:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        
    def transcribe_audio(self, audio_data):
        try:
            # Create a temporary file to store the audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()  # Ensure the file is written to disk
                temp_file.close()  # Explicitly close the file after writing
                
                # Now that the file is closed, we can process it
                result = self.whisper_model.transcribe(temp_file.name)
                
                # Clean up by deleting the temporary file after use
                os.unlink(temp_file.name)
                return result["text"]
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None


def load_resources():
    """Load ASL images and GIFs"""
    images = {}
    gifs = {}
    
    # Load ASL letter images
    for img in os.listdir("images"):
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join("images", img)
            label = os.path.splitext(img)[0].upper()
            images[label] = path
    
    # Load word/phrase GIFs
    for gif in os.listdir("gifs"):
        if gif.lower().endswith('.gif'):
            path = os.path.join("gifs", gif)
            label = os.path.splitext(gif)[0].upper()
            gifs[label] = path
    
    return images, gifs

# def display_asl_sequence(text, images):
#     """Display ASL letter sequence with improved layout"""
#     if not text:
#         return
    
#     text = text.upper()
#     words = text.split()
    
#     for word_idx, word in enumerate(words):
#         if word_idx > 0:
#             st.markdown("<div style='margin: 20px 0;'></div>")
        
#         st.markdown(f"""
#             <div style="background-color: #f0f8ff; 
#                        padding: 10px; 
#                        border-radius: 10px; 
#                        margin-bottom: 10px">
#                 <h3 style="margin: 0; color: #2e7d32; text-align: center">
#                     {word}
#                 </h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         max_chars_per_row = min(len(word), 6)
#         num_rows = math.ceil(len(word) / max_chars_per_row)
        
#         for row in range(num_rows):
#             start_idx = row * max_chars_per_row
#             end_idx = min((row + 1) * max_chars_per_row, len(word))
#             word_segment = word[start_idx:end_idx]
            
#             cols = st.columns(len(word_segment))
#             for idx, char in enumerate(word_segment):
#                 with cols[idx]:
#                     if char.isalpha() and char in images:
#                         img = Image.open(images[char])
#                         img = img.resize((150, 150))
#                         st.image(img, caption=char)
#                     else:
#                         st.write(char)

def audio_to_sign():
    """Handle audio input and conversion to ASL"""
    images, gifs = load_resources()
    audio_processor = AudioProcessor()
    
    tab1, tab2 = st.tabs(["🎤 Voice Input", "📁 File Upload"])
    
    with tab1:
        st.subheader("Speak to See Signs")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎤 Start Recording", key="record"):
                with st.spinner("Listening..."):
                    try:
                        recognizer = sr.Recognizer()
                        with sr.Microphone() as source:
                            st.info("Speak now...")
                            audio = recognizer.listen(source, timeout=5)
                            try:
                                text = recognizer.recognize_google(audio)
                            except sr.UnknownValueError:
                                st.error("Could not understand audio")
                            except sr.RequestError as e:
                                st.error(f"Could not request results; {e}")

                            show_sign_results(text, images, gifs)
                    except Exception as e:
                        st.error("Error recording audio. Please try again.")
    
    with tab2:
        st.subheader("Upload Audio File")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            audio_file = st.file_uploader("Choose an audio file", 
                                        type=["mp3", "wav", "ogg"])
            if audio_file:
                st.audio(audio_file)
                with st.spinner("Processing audio..."):
                    text = audio_processor.transcribe_audio(audio_file.read())
                    if text:
                        show_sign_results(text, images, gifs)

def show_sign_results(text, images, gifs):
    """Display ASL signs for detected text"""
    if not text:
        return
        
    st.success(f"Detected text: {text}")
    
    # Display ASL letter sequence
    st.markdown("""
        <div style="background-color: #e8f5e9; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 20px 0;">
            <h2 style="color: #2e7d32; margin: 0;">ASL Letter Sequence</h2>
        </div>
    """, unsafe_allow_html=True)
    
    display_asl_sequence(text, images)
    
    # Display matching GIF if available
    st.markdown("""
        <div style="background-color: #e8f5e9; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 20px 0;">
            <h2 style="color: #2e7d32; margin: 0;">Word/Phrase Sign</h2>
        </div>
    """, unsafe_allow_html=True)
    
    text = text.upper()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if text in gifs:
            st.image(gifs[text], caption=text, use_container_width=True)
        else:
            matches = difflib.get_close_matches(text, list(gifs.keys()), 
                                              n=1, cutoff=0.8)
            if matches:
                st.image(gifs[matches[0]], caption=matches[0], 
                        use_container_width=True)
            else:
                st.info("No matching sign GIF found for this phrase")

def main():
    st.set_page_config(
        page_title="ASL Communication Hub",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --background: #1a1a1a;
            --secondary-bg: #2d2d2d;
            --accent: #7289da;
            --text: #ffffff;
            --text-secondary: #b0b0b0;
            --success: #43b581;
            --error: #f04747;
            --warning: #faa61a;
        }
        
        /* Global styles */
        .main {
            background-color: var(--background);
            color: var(--text);
        }
        
        .stButton > button {
            background-color: var(--accent);
            color: var(--text);
            border-radius: 20px;
            padding: 15px 32px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #5b6eae;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(114, 137, 218, 0.3);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: var(--secondary-bg);
            padding: 10px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 24px;
            font-size: 18px;
            font-weight: 500;
            color: var(--text);
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 600;
        }
        
        /* Info boxes */
        .info-box {
            background-color: var(--secondary-bg);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 5px solid var(--accent);
            color: var(--text);
        }
        
        /* Success message */
        .success-box {
            background-color: rgba(67, 181, 129, 0.2);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 5px solid var(--success);
            color: var(--text);
        }
        
        /* Word containers */
        .word-container {
            background-color: var(--secondary-bg);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: var(--secondary-bg);
        }
        
        /* Custom components */
        .custom-header {
            background: linear-gradient(135deg, var(--accent), #5b6eae);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .custom-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .custom-subheader {
            color: var(--text-secondary);
            font-size: 1.2em;
            margin-top: 10px;
        }
        
        /* File uploader */
        .uploadedFile {
            background-color: var(--secondary-bg);
            border: 1px solid var(--accent);
            border-radius: 10px;
        }
        
        /* Audio player */
        audio {
            width: 100%;
            border-radius: 10px;
            background-color: var(--secondary-bg);
        }
        
        /* Markdown text */
        .stMarkdown {
            color: var(--text);
        }
        
        /* Separator */
        hr {
            border-color: var(--secondary-bg);
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with dark theme
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #7289da;">🤟 ASL Hub</h1>
        </div>
    """, unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Select Communication Mode",
        ["Sign to Text", "Audio to Sign"],
        format_func=lambda x: "👐 " + x if x == "Sign to Text" else "🗣️ " + x,
        key="mode_selection"
    )

    st.sidebar.markdown("""
        <div style="margin-top: 30px; 
                    padding: 20px; 
                    background-color: rgba(114, 137, 218, 0.1); 
                    border-radius: 10px; 
                    border: 1px solid rgba(114, 137, 218, 0.2);">
            <h4 style="color: #7289da; margin-top: 0;">About This App</h4>
            <p style="color: #b0b0b0;">
                Bridge the communication gap between spoken language and 
                American Sign Language with real-time translation capabilities.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main content with custom header
    st.markdown("""
        <div class="custom-header">
            <h1>ASL Communication Hub</h1>
            <p class="custom-subheader">
                Breaking down communication barriers with real-time ASL translation
            </p>
        </div>
    """, unsafe_allow_html=True)

    if mode == "Sign to Text":
        detector = ASLDetector()
        
        st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">Sign Language Detection</h3>
                <p>Start the detection to convert your sign language gestures into text and speech.</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("✨ Start Sign Detection", key="start_detection"):
                st.session_state.detection_active = True
                sentence = detector.process_camera()
                if sentence:
                    st.markdown(f"""
                        <div class="success-box">
                            <h4 style="margin-top: 0;">Detected Text:</h4>
                            <p style="font-size: 1.2em;">{sentence}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Converting to speech..."):
                        try:
                            tts = gTTS(text=sentence, lang='en')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format="audio/mp3")
                               # os.remove(fp.name)
                        except Exception as e:
                            st.error(f"Error generating audio: {e}")
        
        with col2:
            st.markdown("""
                <div style="background-color: rgba(114, 137, 218, 0.1); 
                           padding: 20px; 
                           border-radius: 10px; 
                           margin-top: 20px;
                           border: 1px solid rgba(114, 137, 218, 0.2);">
                    <h4 style="color: #7289da; margin-top: 0;">Quick Tips:</h4>
                    <ul style="color: #b0b0b0;">
                        <li>Keep your hand within the green box</li>
                        <li>Ensure good lighting</li>
                        <li>Make clear, deliberate signs</li>
                        <li>Take your time between signs</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        audio_to_sign()

def display_asl_sequence(text, images):
    """Display ASL letter sequence with dark theme styling"""
    if not text:
        return
    
    text = text.upper()
    words = text.split()
    
    for word_idx, word in enumerate(words):
        if word_idx > 0:
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="word-container">
                <h3 style="margin: 0; color: #7289da; text-align: center">
                    {word}
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        max_chars_per_row = min(len(word), 6)
        num_rows = math.ceil(len(word) / max_chars_per_row)
        
        for row in range(num_rows):
            start_idx = row * max_chars_per_row
            end_idx = min((row + 1) * max_chars_per_row, len(word))
            word_segment = word[start_idx:end_idx]
            
            cols = st.columns(len(word_segment))
            for idx, char in enumerate(word_segment):
                with cols[idx]:
                    if char.isalpha() and char in images:
                        img = Image.open(images[char])
                        img = img.resize((150, 150))
                        st.image(img, caption=char,use_container_width=True)
                    else:
                        st.write(char)

if __name__ == "__main__":
    main()         