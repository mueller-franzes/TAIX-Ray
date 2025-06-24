# TAIX-Ray Models

This repository provides two trained deep learning models for classifying X-ray images from the TAIX-Ray dataset:

1. Binary Classification Model - Classifies X-ray images into two categories (normal vs. abnormal).

2. Ordinal Classification Model - Predicts severity levels based on ordinal categories.

Please see our paper for a detailed description:  [TAIX-Ray Paper](https://arxiv.org/abs/your-paper-link)


<br>

## How to Use

### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install huggingface_hub
```


### Download 

```python
from huggingface_hub import hf_hub_download


# Download the checkpoint file from Hugging Face Hub
file_path = hf_hub_download(
    repo_id="TLAIM/TAIX-Ray", 
    filename="binary.ckpt",   # binary.ckpt or ordinal.ckpt 
)

# Check if the file has been correctly downloaded
print(f"Checkpoint downloaded to: {file_path}")
```


