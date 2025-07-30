import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINOPredictor:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny"):
        """
        Initialize the Grounding DINO predictor with the specified model.
        
        Args:
            model_id (str): The Hugging Face model ID for Grounding DINO.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
    
    def predict_image(self, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Run prediction on an image with the given text prompt.
        
        Args:
            image (PIL.Image): The input image.
            text_prompt (str): Comma-separated list of objects to detect.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text labels.
            
        Returns:
            tuple: (boxes, scores, labels) where boxes are in [x1, y1, x2, y2] format.
        """
        # Prepare text labels
        text_labels = [[label.strip() for label in text_prompt.split(",")]]
        
        # Prepare inputs
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        
        return boxes, scores, labels