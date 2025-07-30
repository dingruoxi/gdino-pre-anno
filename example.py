import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.grounding_dino import GroundingDINOPredictor
from utils.visualization import draw_boxes_on_image
from utils.annotation_utils import save_annotations

def parse_args():
    parser = argparse.ArgumentParser(description="Example script for Grounding DINO pre-annotation")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="person, car, dog, cat", help="Text prompt for detection")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--format", type=str, default="COCO", choices=["COCO", "PASCAL VOC"], help="Output format")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load image
    image = Image.open(args.image)
    
    # Initialize Grounding DINO predictor
    predictor = GroundingDINOPredictor()
    
    # Run prediction
    print(f"Running prediction with prompt: {args.prompt}")
    boxes, scores, labels = predictor.predict_image(
        image,
        args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    
    # Convert to list of dictionaries
    annotations = []
    for box, score, label in zip(boxes, scores, labels):
        annotations.append({
            "bbox": box.tolist(),  # [x1, y1, x2, y2]
            "score": float(score),
            "label": label
        })
    
    print(f"Found {len(annotations)} objects")
    
    # Create annotations dictionary
    annotations_dict = {args.image: annotations}
    
    # Save annotations
    save_annotations(annotations_dict, args.format, args.output)
    print(f"Saved annotations in {args.format} format to {args.output}")
    
    # Visualize results
    img_np = np.array(image)
    img_with_boxes = draw_boxes_on_image(img_np, annotations)
    
    # Save visualization
    output_image_path = os.path.join(args.output, f"visualization_{os.path.basename(args.image)}")
    plt.figure(figsize=(12, 12))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to {output_image_path}")

if __name__ == "__main__":
    main()