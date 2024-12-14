import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io
import joblib

# Load the saved CNN model
model = tf.keras.models.load_model('voice_verification_cnn.keras')

# Load normalization values
normalization_values = joblib.load("normalization_values.pkl")
MEAN_VAL = normalization_values["mean"]
STD_VAL = normalization_values["std"]

# Parameters used during training
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION)

def preprocess_audio(file_data):
    """Preprocess audio file for the CNN model."""
    y, sr = librosa.load(io.BytesIO(file_data), sr=SAMPLE_RATE, mono=True)

    # Ensure fixed length
    if len(y) > SAMPLES_PER_CLIP:
        y = y[:SAMPLES_PER_CLIP]
    else:
        padding = SAMPLES_PER_CLIP - len(y)
        y = np.concatenate([y, np.zeros(padding)])

    # Extract log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    log_mel_spec = (log_mel_spec - MEAN_VAL) / (STD_VAL + 1e-9)

    # Add channel dimension for CNN input
    log_mel_spec = log_mel_spec[..., np.newaxis]  # (n_mels, time_steps, 1)

    # Add batch dimension
    log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, n_mels, time_steps, 1)
    return log_mel_spec

def predict_audio(file_data):
    """Make a prediction on the uploaded audio file."""
    features = preprocess_audio(file_data)
    prediction = model.predict(features)[0]  # e.g., [0.8, 0.2]
    label = "Real" if np.argmax(prediction) == 0 else "Fake"
    confidence = prediction[np.argmax(prediction)]


    print(f"Prediction: {prediction}")
    print(f"Prediction shape: {prediction.shape}")
    print(np.argmax(prediction))
    print(confidence)

    return label, confidence

st.title("Voice Verification System")
st.write("Upload a .wav file to verify if it is 'Real' or 'Fake'.")

uploaded_file = st.file_uploader("Upload a file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    label, confidence = predict_audio(uploaded_file.read())
    if label == 'Real':
        st.success(f"Prediction: {label} (Confidence: {100 * confidence:.2f} %)")
    else:
        st.error(f"Prediction: {label} (Confidence: {100 * confidence:.2f} %)")
