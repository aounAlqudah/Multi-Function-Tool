from transformers import WhisperProcessor, WhisperForConditionalGeneration
import json
import torch
import librosa


# Load fine-tuned Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained("./resultsAudio")
whisper_model = WhisperForConditionalGeneration.from_pretrained("./resultsAudio")
whisper_model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model.to(device)

# Words to highlight in the text
words_to_highlight = [
    "Risk", "cost", "delay", "started", "finished", "in progress",
    "caution", "situation", "time overrun", "cost overrun"
]

# Highlight specific words in text
def highlight_words(text, words_to_highlight):
    for word in words_to_highlight:
        text = text.replace(word, f'<span class="highlight">{word}</span>')
    return text


def load_audio(file_path, sampling_rate=16000):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio

# Function to transcribe audio
def transcribe_audio(file_path):
    audio = load_audio(file_path)
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        generated_ids = whisper_model.generate(inputs)
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

