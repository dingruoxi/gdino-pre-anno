import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from utils.grounding_dino import GroundingDINOPredictor
from utils.visualization import draw_boxes_on_image

def test_with_sample_image():
    """
    Test the Grounding DINO model with a sample image from the internet.
    """
    # URL of a sample image (COCO image)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    try:
        # Download the image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # Initialize the predictor
        predictor = GroundingDINOPredictor()
        
        # Run prediction
        text_prompt = "cat, remote"
        print(f"Running prediction with prompt: {text_prompt}")
        
        boxes, scores, labels = predictor.predict_image(
            image,
            text_prompt,
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        # Convert to list of dictionaries
        annotations = []
        for box, score, label in zip(boxes, scores, labels):
            annotations.append({
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "score": float(score),
                "label": label
            })
        
        print(f"Found {len(annotations)} objects:")
        for ann in annotations:
            print(f"  {ann['label']}: {ann['score']:.2f} at {[int(x) for x in ann['bbox']]}")
        
        # Visualize results
        img_np = np.array(image)
        img_with_boxes = draw_boxes_on_image(img_np, annotations)
        
        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.title(f"Detected objects with prompt: {text_prompt}")
        plt.show()
        
        return True
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

def main():
    print("Testing Grounding DINO model...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run the test
    success = test_with_sample_image()
    
    if success:
        print("\nTest completed successfully! The model is working correctly.")
    else:
        print("\nTest failed. Please check the error message above.")

if __name__ == "__main__":
    main()