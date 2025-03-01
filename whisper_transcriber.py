import streamlit as st
import whisper
import os

# Set FFmpeg path manually (this helps if FFmpeg isn't detected)
os.environ["PATH"] += os.pathsep + "/usr/local/bin"

def save_uploaded_file(uploaded_file):
    """Save the uploaded file temporarily."""
    file_path = os.path.join("temp_audio.mp3")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def transcribe_audio(file_path):
    """Transcribe audio using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# Streamlit App UI
st.title("Whisper MP3 Transcriber")
st.write("Upload an MP3 file, and it will be transcribed using OpenAI's Whisper.")

uploaded_file = st.file_uploader("Upload MP3 File", type=["mp3"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    file_path = save_uploaded_file(uploaded_file)
    
    st.write("Transcribing...")
    transcription = transcribe_audio(file_path)
    
    st.subheader("Transcription:")
    st.text_area("", transcription, height=300)
    
    # Provide a download button
    st.download_button("Download Transcription", transcription, "transcription.txt", "text/plain")
    
    # Cleanup
    os.remove(file_path)
