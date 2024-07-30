from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load dataset
dataset = load_dataset('json', data_files={'train': 'train.json'})

# Print dataset structure for debugging
print(dataset)

# Add audio feature to the dataset
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Preprocess dataset
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def preprocess_function(examples):
    # Debugging: Print examples to understand their structure
    print(examples)
    
    audio = examples["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    
    # Tokenize the text
    labels = processor.tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=448).input_ids
    
    return {"input_features": inputs.squeeze(), "labels": labels.squeeze()}

# Apply the preprocess function to the dataset
dataset = dataset.map(preprocess_function, remove_columns=["audio", "text"])

# Save processed dataset
dataset.save_to_disk('processed_dataset')
