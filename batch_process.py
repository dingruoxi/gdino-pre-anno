import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.grounding_dino import GroundingDINOPredictor
from utils.visualization import draw_boxes_on_image
from utils.annotation_utils import save_annotations

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images with Grounding DINO")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for annotations and visualizations")
    parser.add_argument("--prompt", type=str, default="person, car, dog, cat", help="Text prompt for detection")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--format", type=str, default="COCO", choices=["COCO", "PASCAL VOC"], help="Output format")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get list of image files
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for file in os.listdir(args.input_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(args.input_dir, file))
    
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize Grounding DINO predictor
    predictor = GroundingDINOPredictor()
    
    # Process each image
    annotations_dict = {}
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = Image.open(image_path)
            
            # Run prediction
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
            
            # Store annotations
            annotations_dict[image_path] = annotations
            
            # Generate visualization if requested
            if args.visualize:
                img_np = np.array(image)
                img_with_boxes = draw_boxes_on_image(img_np, annotations)
                
                # Save visualization
                output_image_path = os.path.join(vis_dir, f"vis_{os.path.basename(image_path)}")
                plt.figure(figsize=(12, 12))
                plt.imshow(img_with_boxes)
                plt.axis('off')
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Save annotations
    save_annotations(annotations_dict, args.format, args.output_dir)
    print(f"Saved annotations in {args.format} format to {args.output_dir}")
    
    if args.visualize:
        print(f"Saved visualizations to {vis_dir}")

if __name__ == "__main__":
    main()