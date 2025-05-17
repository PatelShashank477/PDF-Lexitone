# 📚 PDF-Lexitone: Transform PDFs into Audio Summaries

**PDF-Lexitone** is an innovative Python application that converts PDF documents into concise audio summaries. Leveraging advanced natural language processing techniques and text-to-speech synthesis, it provides an accessible way to consume lengthy documents audibly.

---

## 🚀 Features

* **PDF Text Extraction**: Upload and extract text from PDF files.
* **Summarization Techniques**:

  * *Abstractive Summarization*: Utilizes the T5 Transformer model to generate human-like summaries.
  * *Extractive Summarization*: Employs TF-IDF to select key sentences from the text.
* **Text-to-Speech Conversion**:

  * Standard TTS using Coqui TTS.
  * Character-based voices.
  * Voice cloning with YourTTS.
* **Interactive Web Interface**: Built with Streamlit for user-friendly interaction.
* **Audio Playback and Download**: Listen to or download the generated audio summaries.

---

## 🗂️ Project Structure

```
PDF-Lexitone/
├── audio_samples/           # Directory to store generated audio files
├── data/                    # Temporary data or summary outputs
├── nltk_data/               # NLTK tokenizer resources
│   └── tokenizers/
├── main1.py                 # Core logic: summarization and audio generation
├── test_streamlit.py        # Streamlit app for user interface
└── README.md                # Project documentation
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.7 or higher
* pip package manager

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/PatelShashank477/PDF-Lexitone.git
   cd PDF-Lexitone
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   If a `requirements.txt` file is present:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, manually install the required packages:

   ```bash
   pip install streamlit PyPDF2 nltk TTS transformers torch
   ```

4. **Download NLTK Tokenizer Data**:

   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## 🧪 Usage

1. **Launch the Streamlit Application**:

   ```bash
   streamlit run test_streamlit.py
   ```

2. **Interact with the Application**:

   * Upload a PDF file.
   * Choose the summarization method: *Abstractive (T5)* or *Extractive (TF-IDF)*.
   * Select the desired voice output: *Standard TTS*, *Character Voice*, or *YourTTS Clone*.
   * Play or download the generated audio summary.

---

## 🧠 Technologies Used

| Technology     | Description                           |
| -------------- | ------------------------------------- |
| Python         | Programming language                  |
| Streamlit      | Web application framework             |
| PyPDF2         | PDF text extraction                   |
| NLTK           | Natural language processing           |
| T5 Transformer | Abstractive summarization model       |
| TF-IDF         | Extractive summarization technique    |
| Coqui TTS      | Text-to-speech synthesis              |
| YourTTS        | Voice cloning for personalized output |

---

## 🎧 Audio Output Options

| Mode         | Description                                    |
| ------------ | ---------------------------------------------- |
| Standard TTS | Converts text to speech using default settings |
| Character    | Uses predefined synthetic character voices     |
| YourTTS      | Clones and uses a specific speaker's voice     |

*All generated audio files are saved in the `audio_samples/` directory.*

---

## ❓ Troubleshooting

* **Streamlit App Not Launching**:

  * Ensure all dependencies are installed correctly.
  * Verify that you're using Python 3.7 or higher.

* **NLTK Errors**:

  * Make sure you've downloaded the necessary tokenizer data:

    ```python
    import nltk
    nltk.download('punkt')
    ```

* **Audio Not Playing**:

  * Check for errors related to `torch` or `TTS` installations.
  * Ensure your system supports audio playback.

---


## 📌 Future Enhancements

* [ ] Support for multiple languages in summarization and TTS.
* [ ] Ability to upload and process DOCX and TXT files.
* [ ] Export summaries as text or PDF documents.
* [ ] User authentication and session management for saving history.

---

