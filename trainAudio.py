from transformers import Trainer, TrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk, DatasetDict
import torch

# Load processed dataset
dataset = load_from_disk('processed_dataset')

# Split the dataset into train and validation sets
split_dataset = dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Set the device to CPU
device = torch.device("cpu")
model.to(device)

# Define training arguments with reduced batch size and gradient accumulation
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,   # Reduced batch size
    eval_strategy="steps",
    num_train_epochs=3,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=2e-5,
    save_total_limit=2,
    gradient_accumulation_steps=4,  # Simulate larger batch size
    no_cuda=True  # Disable CUDA
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train the model
trainer.train()

# Save the fine-tuned model and processor
model.save_pretrained("./results")
processor.save_pretrained("./results")
