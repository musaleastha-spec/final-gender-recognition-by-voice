import streamlit as st
import joblib
import librosa
import numpy as np

# Set page configuration
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️", layout="centered")

# Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_gender_classifier.pkl")
    scaler = joblib.load("voice_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    artifacts_loaded = False

# Prediction logic
def classify_voice(audio_path):
    try:
        # Avoid resampy dependency by using default/soxr resampler
        y, sr = librosa.load(audio_path, duration=3.0)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features_list = [mean_pitch, spec_centroid, zcr] + list(mfcc_mean)
        features_array = np.array(features_list).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        # Retrieve prediction probabilities if available, else fall back to hard prediction
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_features)[0]
            # Assuming class mapping: 1 -> Male, 0 -> Female
            classes = getattr(model, "classes_", [0, 1])
            res = {}
            for cls, prob in zip(classes, probs):
                gender_str = "Male" if cls == 1 else "Female"
                res[gender_str] = prob
            return res
        else:
            pred = model.predict(scaled_features)[0]
            gender_str = "Male" if pred == 1 else "Female"
            return {gender_str: 1.0}

    except Exception as e:
        return f"Error processing audio: {str(e)}"

# App UI
st.title("🎙️ Voice Gender Recognition AI")
st.write("Upload a `.wav` audio file to classify the speaker's gender.")

if artifacts_loaded:
    audio_file = st.file_uploader("Choose a WAV file", type=["wav"])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        if st.button("Analyze Voice", type="primary"):
            with st.spinner("Analyzing your audio sample..."):
                temp_filename = "temp_audio.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                result = classify_voice(temp_filename)
                
                if isinstance(result, dict):
                    st.success("Analysis Complete!")
                    st.subheader("Prediction Results")
                    for gender_class, probability in result.items():
                        label_text = "Male" if str(gender_class) in ["1", "Male"] else "Female"
                        st.write(f"**{label_text}**: {probability * 100:.2f}%")
                        st.progress(float(probability))
                else:
                    st.error(result)
