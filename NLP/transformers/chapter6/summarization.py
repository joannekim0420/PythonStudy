from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", version="3.0.0")
print(f"특성:{dataset['train'].column_names}")
