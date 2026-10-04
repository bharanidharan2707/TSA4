import streamlit as st
import torch
import soundfile as sf
import librosa
from transformers import pipeline
import io

st.set_page_config(page_title="Speech Emotion Detection", page_icon="🎤", layout="centered")

@st.cache_resource
def load_model():
    # Using a pre-trained model from Hugging Face that predicts neu, hap, ang, sad
    # The 'superb/wav2vec2-base-superb-er' model predicts 4 emotion classes.
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    return classifier

st.title("🎤 Speech Emotion Detection")
st.write("Upload an audio file or record your voice to detect the emotion (Happy, Sad, Angry, Neutral).")

# Load model
try:
    with st.spinner("Loading AI model (this may take a minute on first run)..."):
        classifier = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Allow uploading
st.subheader("Upload Audio")
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, etc.)", type=['wav', 'mp3', 'ogg', 'flac'])

# Allow recording
st.subheader("Or Record Your Voice")
audio_value = st.audio_input("Record a voice message")

audio_to_process = None
if audio_value is not None:
    audio_to_process = audio_value
    st.success("Audio recorded successfully!")
elif uploaded_file is not None:
    audio_to_process = uploaded_file
    st.success("File uploaded successfully!")

if audio_to_process is not None:
    st.audio(audio_to_process)
    
    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing emotion..."):
            try:
                # Read the audio file
                # Use sf.read which can handle file-like objects
                speech, samplerate = sf.read(audio_to_process)
                
                # Resample to 16kHz if needed (wav2vec2 requires 16kHz)
                if samplerate != 16000:
                    speech = librosa.resample(y=speech.T, orig_sr=samplerate, target_sr=16000).T
                    samplerate = 16000
                
                # Ensure mono by averaging channels if stereo
                if len(speech.shape) > 1 and speech.shape[1] > 1:
                    speech = speech.mean(axis=1)

                # Convert to numpy array of float32 if not already
                speech = speech.astype("float32")

                # Predict
                predictions = classifier({"sampling_rate": samplerate, "raw": speech})
                
                # Extract top prediction
                top_prediction = predictions[0]
                emotion = top_prediction['label'].capitalize()
                score = top_prediction['score']
                
                # Map short labels to full words
                emotion_map = {
                    "Neu": "Neutral",
                    "Hap": "Happy",
                    "Ang": "Angry",
                    "Sad": "Sad"
                }
                
                # E.g., 'neu' -> 'Neutral'
                mapped_emotion = emotion_map.get(emotion[:3].capitalize(), emotion)
                
                # Display results
                st.success(f"### Detected Emotion: **{mapped_emotion}**")
                st.write(f"Confidence: {score:.2%}")
                
                st.write("---")
                st.write("**All probabilities:**")
                for p in predictions:
                    label = emotion_map.get(p['label'].capitalize()[:3], p['label'].capitalize())
                    st.progress(float(p['score']), text=f"{label}: {p['score']:.2%}")

            except Exception as e:
                st.error(f"Error processing audio: {e}")
                st.info("Try uploading a standard WAV file if you encounter issues.")
