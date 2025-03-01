import streamlit as st
import whisper
import os
import ffmpeg  # Import ffmpeg-python to handle audio

def save_uploaded_file(uploaded_file):
    """Save the uploaded file temporarily."""
    file_path = "temp_audio.mp3"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def transcribe_audio(file_path):
    """Transcribe audio using Whisper."""
    model = whisper.load_model("base")
    
    # Convert audio using ffmpeg-python (ensures compatibility)
    input_audio = ffmpeg.input(file_path)
    output_audio = ffmpeg.output(input_audio, "converted_audio.mp3", format="mp3")
    ffmpeg.run(output_audio)

    result = model.transcribe("converted_audio.mp3")
    
    # Cleanup converted file
    os.remove("converted_audio.mp3")

    return result["text"]

# Streamlit UI
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
