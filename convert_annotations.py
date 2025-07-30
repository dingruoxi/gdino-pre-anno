import os
import argparse
from utils.annotation_utils import load_annotations, save_annotations

def parse_args():
    parser = argparse.ArgumentParser(description="Convert annotations between different formats")
    parser.add_argument("--input", type=str, required=True, help="Input annotation file or directory")
    parser.add_argument("--input-format", type=str, required=True, choices=["COCO", "PASCAL VOC"], help="Input format")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--output-format", type=str, required=True, choices=["COCO", "PASCAL VOC"], help="Output format")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {args.input} in {args.input_format} format...")
    annotations = load_annotations(args.input, args.input_format)
    
    # Save annotations in the output format
    print(f"Saving annotations to {args.output} in {args.output_format} format...")
    save_annotations(annotations, args.output_format, args.output)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()