import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.grounding_dino import GroundingDINOPredictor
from utils.visualization import draw_boxes_on_image
from utils.annotation_utils import save_annotations, load_annotations

def parse_args():
    parser = argparse.ArgumentParser(description="Command-line interface for annotation tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Pre-annotate command
    annotate_parser = subparsers.add_parser("annotate", help="Pre-annotate images with Grounding DINO")
    annotate_parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    annotate_parser.add_argument("--output", type=str, required=True, help="Output directory for annotations")
    annotate_parser.add_argument("--prompt", type=str, default="person, car, dog, cat", help="Text prompt for detection")
    annotate_parser.add_argument("--box-threshold", type=float, default=0.35, help="Box threshold")
    annotate_parser.add_argument("--text-threshold", type=float, default=0.25, help="Text threshold")
    annotate_parser.add_argument("--format", type=str, default="COCO", choices=["COCO", "PASCAL VOC"], help="Output format")
    annotate_parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert annotations between formats")
    convert_parser.add_argument("--input", type=str, required=True, help="Input annotation file or directory")
    convert_parser.add_argument("--input-format", type=str, required=True, choices=["COCO", "PASCAL VOC"], help="Input format")
    convert_parser.add_argument("--output", type=str, required=True, help="Output directory")
    convert_parser.add_argument("--output-format", type=str, required=True, choices=["COCO", "PASCAL VOC"], help="Output format")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize annotations")
    visualize_parser.add_argument("--image", type=str, required=True, help="Input image")
    visualize_parser.add_argument("--annotations", type=str, required=True, help="Annotation file or directory")
    visualize_parser.add_argument("--format", type=str, required=True, choices=["COCO", "PASCAL VOC"], help="Annotation format")
    visualize_parser.add_argument("--output", type=str, help="Output image file (if not specified, display on screen)")
    
    return parser.parse_args()

def annotate(args):
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file mode
        image_files = [args.input]
    elif os.path.isdir(args.input):
        # Directory mode
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for file in os.listdir(args.input):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(args.input, file))
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    if not image_files:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.output, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize Grounding DINO predictor
    predictor = GroundingDINOPredictor()
    
    # Process each image
    annotations_dict = {}
    for image_path in image_files:
        try:
            print(f"Processing {image_path}...")
            
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
            
            print(f"  Found {len(annotations)} objects")
            
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
                print(f"  Saved visualization to {output_image_path}")
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Save annotations
    save_annotations(annotations_dict, args.format, args.output)
    print(f"Saved annotations in {args.format} format to {args.output}")

def convert(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {args.input} in {args.input_format} format...")
    annotations = load_annotations(args.input, args.input_format)
    
    # Save annotations in the output format
    print(f"Saving annotations to {args.output} in {args.output_format} format...")
    save_annotations(annotations, args.output_format, args.output)
    
    print("Conversion completed successfully!")

def visualize(args):
    # Load image
    try:
        image = Image.open(args.image)
        img_np = np.array(image)
    except Exception as e:
        print(f"Error loading image {args.image}: {str(e)}")
        return
    
    # Load annotations
    try:
        annotations_dict = load_annotations(args.annotations, args.format)
        
        # Find annotations for this image
        annotations = None
        for image_path, anns in annotations_dict.items():
            if os.path.basename(image_path) == os.path.basename(args.image):
                annotations = anns
                break
        
        if annotations is None:
            print(f"No annotations found for {args.image}")
            return
        
        print(f"Found {len(annotations)} annotations for {args.image}")
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        return
    
    # Draw boxes on image
    img_with_boxes = draw_boxes_on_image(img_np, annotations)
    
    # Save or display the visualization
    if args.output:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved visualization to {args.output}")
    else:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.title(f"Annotations for {os.path.basename(args.image)}")
        plt.show()

def main():
    args = parse_args()
    
    if args.command == "annotate":
        annotate(args)
    elif args.command == "convert":
        convert(args)
    elif args.command == "visualize":
        visualize(args)
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == "__main__":
    main()