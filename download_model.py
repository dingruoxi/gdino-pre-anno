import os
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def download_model(model_id="IDEA-Research/grounding-dino-tiny", output_dir="models"):
    """
    Download the Grounding DINO model for offline use.
    
    Args:
        model_id (str): The Hugging Face model ID for Grounding DINO.
        output_dir (str): Directory to save the model files.
    """
    print(f"Downloading model {model_id}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    
    # Save processor and model
    processor.save_pretrained(os.path.join(output_dir, os.path.basename(model_id)))
    model.save_pretrained(os.path.join(output_dir, os.path.basename(model_id)))
    
    print(f"Model saved to {os.path.join(output_dir, os.path.basename(model_id))}")

if __name__ == "__main__":
    # Download the tiny model by default
    download_model()
    
    # Uncomment to download other model variants
    # download_model("IDEA-Research/grounding-dino-base")
    # download_model("IDEA-Research/grounding-dino-b")