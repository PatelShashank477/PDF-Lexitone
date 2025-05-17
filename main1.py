import os
import streamlit as st
import torch
import base64
import nltk
import numpy as np
import io
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from gtts import gTTS
import tempfile
import edge_tts

# Set page config FIRST
st.set_page_config(layout="wide", page_title="Document Summarizer")

st.title("📄 Document Summarization App")
st.markdown("Upload a PDF file and select the summarization method to generate a concise summary.")

# Check if required libraries are installed
try:
    import librosa
    from TTS.api import TTS
except ImportError:
    st.error("Required libraries for voice cloning are not installed. Please run: pip install TTS librosa")
    st.stop()

# Create directory for uploaded files
if not os.path.exists("data"):
    os.makedirs("data")

# Create directory for audio samples
if not os.path.exists("audio_samples"):
    os.makedirs("audio_samples")

# Define available voice characters
VOICE_OPTIONS = {
    "Female (US)": "en-US-JennyNeural",
    "Male (US)": "en-US-GuyNeural",
    "Female (UK)": "en-GB-SoniaNeural",
    "Male (UK)": "en-GB-RyanNeural",
    "Female (Australia)": "en-AU-NatashaNeural",
    "Male (Australia)": "en-AU-WilliamNeural",
    "Female (India)": "en-IN-NeerjaNeural",
    "Male (India)": "en-IN-PrabhatNeural",
    "Female (Germany)": "de-DE-KatjaNeural",
    "Male (Germany)": "de-DE-ConradNeural",
    "Female (France)": "fr-FR-DeniseNeural",
    "Male (France)": "fr-FR-HenriNeural",
    "Female (Japan)": "ja-JP-NanamiNeural",
    "Male (Japan)": "ja-JP-KeitaNeural",
    "Female (Spain)": "es-ES-ElviraNeural",
    "Male (Spain)": "es-ES-AlvaroNeural"
}

with st.spinner("Loading models... This may take a minute..."):
    # Download required NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    
    # Initialize TTS model for voice cloning
    @st.cache_resource
    def load_tts_model():
        try:
            # Initialize TTS with the voice cloning model (YourTTS)
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            return tts
        except Exception as e:
            st.error(f"Error loading TTS model: {str(e)}")
            return None

    # Load TTS model
    tts_model = load_tts_model()
    if tts_model is None:
        st.error("Failed to load voice cloning model. Please check your installation.")

# Extract and split PDF content
@st.cache_data
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# Display PDF in Streamlit
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, device_map='auto', torch_dtype=torch.float32
    )
    return tokenizer, base_model

# Load model
tokenizer, base_model = load_model()

# T5 summarizer function
def summarize_t5(text, max_input_length=1800):
    if len(text) < 100:
        return "Text too short to summarize effectively."
    
    try:
        pipe_sum = pipeline(
            "summarization",
            model=base_model,
            tokenizer=tokenizer,
            max_length=500,
            min_length=50,
        )
        summary = pipe_sum(text[:max_input_length])
        return summary[0]["summary_text"]
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return "Error generating summary. Try with a different text or method."

# TF-IDF summarizer function
def summarize_tfidf(text, top_n=5):
    sentences = sent_tokenize(text)
    
    if len(sentences) <= top_n:
        return text
        
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.mean(tfidf_matrix.toarray(), axis=1)
        top_sentence_indices = scores.argsort()[-top_n:][::-1]
        top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        return " ".join(top_sentences)
    except Exception as e:
        st.error(f"TF-IDF error: {str(e)}")
        return "Error generating summary. Try with a different text or method."

# Function to generate audio from text using Edge TTS (with character voices)
@st.cache_data
def generate_character_audio(text, voice="en-US-JennyNeural"):
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name
        
        # Use edge_tts to generate audio with the selected voice
        async def _generate_audio():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_path)
        
        # Run the async function to generate audio
        import asyncio
        asyncio.run(_generate_audio())
        
        # Read the audio file
        with open(temp_path, "rb") as audio_file:
            audio_bytes = io.BytesIO(audio_file.read())
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return audio_bytes
    except Exception as e:
        st.error(f"Character audio generation error: {str(e)}")
        return None

# Function to generate audio based on a reference sample using YourTTS
def generate_voice_cloned_audio(text, reference_audio_path):
    try:
        if tts_model is None:
            st.error("Voice cloning model is not available")
            return None

        # Create an output file for the synthesized audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            out_path = tmp_out.name
            
        # Use the TTS model to clone the voice and generate audio
        st.info("Cloning voice from sample...")
        
        # YourTTS model requires speaker embedding from reference audio
        tts_model.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=reference_audio_path,
            language="en"
        )
            
        # Read the output audio file
        with open(out_path, "rb") as audio_file:
            audio_bytes = io.BytesIO(audio_file.read())
        
        # Clean up temporary file
        os.unlink(out_path)
        
        return audio_bytes
            
    except Exception as e:
        st.error(f"Voice cloning error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Fallback function to generate audio using gTTS
@st.cache_data
def generate_gtts_audio(text, lang='en'):
    try:
        audio_bytes = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"gTTS audio generation error: {str(e)}")
        return None

st.success("Models loaded successfully!")

# UI elements
with st.sidebar:
    st.header("Settings")
    algo = st.radio("Choose Summarization Method:", 
                   ["T5 (Abstractive)", "TF-IDF (Extractive)"], 
                   help="T5 creates new text, TF-IDF extracts important sentences")
    
    if algo == "TF-IDF (Extractive)":
        sentence_count = st.slider("Number of sentences to extract:", 3, 10, 5)
    else:
        sentence_count = 5
    
    st.subheader("Audio Settings")
    audio_engine = st.radio("Text-to-Speech Engine:", 
                          ["Voice Cloning", "Character Voices", "Standard TTS"],
                          help="Voice Cloning uses your uploaded sample to clone the voice")
    
    if audio_engine == "Character Voices":
        voice_character = st.selectbox("Select Voice Character:", 
                                     list(VOICE_OPTIONS.keys()),
                                     index=0)
        voice_id = VOICE_OPTIONS[voice_character]
    elif audio_engine == "Standard TTS":
        audio_lang = st.selectbox("Audio Language", 
                              ["en", "es", "fr", "de", "it", "pt", "ru", "zh-CN", "ja", "hi"],
                              index=0,
                              help="Select language for text-to-speech")
    else:  # Voice Cloning
        st.info("Upload a sample audio file below to clone the voice")
        st.warning("For best results, upload a clear voice sample with minimal background noise")
        sample_audio = st.file_uploader("Upload Voice Sample (WAV, MP3)", type=["wav", "mp3"])
        if sample_audio:
            st.audio(sample_audio, format=f"audio/{sample_audio.name.split('.')[-1]}")
            sample_path = os.path.join("audio_samples", f"sample_{int(time.time())}.{sample_audio.name.split('.')[-1]}")
            with open(sample_path, "wb") as f:
                f.write(sample_audio.getbuffer())
            st.session_state["sample_path"] = sample_path
            st.success("Voice sample uploaded successfully!")
            st.info("This voice will be used for generating the summary audio")

# Main content
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data", f"{uploaded_file.name}")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Generate Summary"):
        with st.spinner("Processing document..."):
            try:
                text = file_preprocessing(file_path)
                
                if not text.strip():
                    st.error("Could not extract text from this PDF.")
                else:
                    if algo == "T5 (Abstractive)":
                        summary = summarize_t5(text)
                    else:
                        summary = summarize_tfidf(text, top_n=sentence_count)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📘 Uploaded PDF")
                        displayPDF(file_path)

                    with col2:
                        st.subheader("🧠 Generated Summary")
                        st.success(summary)

                        original_words = len(text.split())
                        summary_words = len(summary.split())
                        compression = round((1 - summary_words / original_words) * 100) if original_words > 0 else 0
                        
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        metrics_col1.metric("Original Length", f"{original_words} words")
                        metrics_col2.metric("Summary Length", f"{summary_words} words")
                        metrics_col3.metric("Compression", f"{compression}%")

                        st.download_button(
                            label="📥 Download Summary Text",
                            data=summary,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                            mime="text/plain"
                        )
                        
                        # Generate audio for the summary
                        with st.spinner("Generating audio..."):
                            if audio_engine == "Voice Cloning":
                                if "sample_path" in st.session_state:
                                    with st.spinner("Cloning voice from sample and generating audio..."):
                                        audio_bytes = generate_voice_cloned_audio(
                                            summary, st.session_state["sample_path"]
                                        )
                                    voice_info = "Voice cloned from your audio sample"
                                else:
                                    st.warning("No voice sample provided. Using default voice.")
                                    audio_bytes = generate_character_audio(summary)
                                    voice_info = "Default voice (no sample provided)"
                            elif audio_engine == "Character Voices":
                                audio_bytes = generate_character_audio(summary, voice=voice_id)
                                voice_info = f"Voice: {voice_character} ({voice_id})"
                            else:
                                audio_bytes = generate_gtts_audio(summary, lang=audio_lang)
                                voice_info = f"Language: {audio_lang}"
                            
                            if audio_bytes:
                                st.subheader("🔊 Audio Summary")
                                st.caption(voice_info)
                                st.audio(audio_bytes, format="audio/mp3")
                                
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=audio_bytes,
                                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.mp3",
                                    mime="audio/mp3"
                                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Details: {type(e).__name__}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
    if st.checkbox("Delete uploaded files after processing", value=True):
        try:
            # Delete PDF file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete sample audio if exists
            if "sample_path" in st.session_state and os.path.exists(st.session_state["sample_path"]):
                os.remove(st.session_state["sample_path"])
                del st.session_state["sample_path"]
                
            st.success("Files deleted successfully")
        except Exception as e:
            st.error(f"Error deleting files: {str(e)}")
