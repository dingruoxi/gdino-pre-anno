import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import datetime
from PIL import Image

def save_annotations(annotations_dict, format_type, output_dir):
    """
    Save annotations in the specified format.
    
    Args:
        annotations_dict (dict): Dictionary mapping image paths to annotations.
        format_type (str): Format to save annotations in ('COCO' or 'PASCAL VOC').
        output_dir (str): Directory to save annotation files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type == "COCO":
        save_coco_format(annotations_dict, output_dir)
    elif format_type == "PASCAL VOC":
        save_pascal_voc_format(annotations_dict, output_dir)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def save_coco_format(annotations_dict, output_dir):
    """
    Save annotations in COCO format.
    
    Args:
        annotations_dict (dict): Dictionary mapping image paths to annotations.
        output_dir (str): Directory to save the COCO JSON file.
    """
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "Dataset created with Annotation Tool",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Keep track of categories
    categories = {}
    category_id = 1
    
    # Keep track of annotation ID
    annotation_id = 1
    
    # Process each image and its annotations
    for image_id, (image_path, image_annotations) in enumerate(annotations_dict.items(), 1):
        # Get image information
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        # Add image to COCO format
        coco_data["images"].append({
            "id": image_id,
            "license": 1,
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "date_captured": ""
        })
        
        # Process annotations for this image
        for ann in image_annotations:
            label = ann["label"]
            
            # Add category if it doesn't exist
            if label not in categories:
                categories[label] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": label,
                    "supercategory": "none"
                })
                category_id += 1
            
            # Get bounding box
            x1, y1, x2, y2 = ann["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # Add annotation to COCO format
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": categories[label],
                "bbox": [x1, y1, width, height],  # COCO format: [x, y, width, height]
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0,
                "score": ann.get("score", 1.0)
            })
            
            annotation_id += 1
    
    # Save COCO JSON file
    output_file = os.path.join(output_dir, "annotations.json")
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Saved COCO annotations to {output_file}")

def save_pascal_voc_format(annotations_dict, output_dir):
    """
    Save annotations in PASCAL VOC format.
    
    Args:
        annotations_dict (dict): Dictionary mapping image paths to annotations.
        output_dir (str): Directory to save the XML files.
    """
    # Create annotations directory
    annotations_dir = os.path.join(output_dir, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Process each image and its annotations
    for image_path, image_annotations in annotations_dict.items():
        # Get image information
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                depth = 3  # Assume RGB
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        # Create XML structure
        annotation = ET.Element("annotation")
        
        # Add basic image information
        ET.SubElement(annotation, "folder").text = os.path.dirname(image_path)
        ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
        ET.SubElement(annotation, "path").text = image_path
        
        # Add source information
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"
        
        # Add size information
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        
        # Add segmented information
        ET.SubElement(annotation, "segmented").text = "0"
        
        # Add object annotations
        for ann in image_annotations:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = ann["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            # Add bounding box
            bbox = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = [int(coord) for coord in ann["bbox"]]
            ET.SubElement(bbox, "xmin").text = str(x1)
            ET.SubElement(bbox, "ymin").text = str(y1)
            ET.SubElement(bbox, "xmax").text = str(x2)
            ET.SubElement(bbox, "ymax").text = str(y2)
            
            # Add confidence score as an extra field
            ET.SubElement(obj, "confidence").text = str(ann.get("score", 1.0))
        
        # Convert to pretty XML string
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        
        # Save XML file
        output_file = os.path.join(annotations_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.xml")
        with open(output_file, "w") as f:
            f.write(xml_str)
    
    print(f"Saved PASCAL VOC annotations to {annotations_dir}")

def load_annotations(file_path, format_type):
    """
    Load annotations from a file.
    
    Args:
        file_path (str): Path to the annotation file.
        format_type (str): Format of the annotation file ('COCO' or 'PASCAL VOC').
        
    Returns:
        dict: Dictionary mapping image paths to annotations.
    """
    if format_type == "COCO":
        return load_coco_format(file_path)
    elif format_type == "PASCAL VOC":
        return load_pascal_voc_format(file_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def load_coco_format(file_path):
    """
    Load annotations from a COCO format JSON file.
    
    Args:
        file_path (str): Path to the COCO JSON file.
        
    Returns:
        dict: Dictionary mapping image paths to annotations.
    """
    with open(file_path, "r") as f:
        coco_data = json.load(f)
    
    # Create a mapping from image ID to file name
    image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # Create a mapping from category ID to category name
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        
        # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x + w, y + h]
        
        annotations_by_image[image_id].append({
            "bbox": bbox,
            "score": ann.get("score", 1.0),
            "label": category_id_to_name[ann["category_id"]]
        })
    
    # Map file names to annotations
    result = {}
    for image_id, annotations in annotations_by_image.items():
        file_name = image_id_to_file[image_id]
        result[file_name] = annotations
    
    return result

def load_pascal_voc_format(directory):
    """
    Load annotations from PASCAL VOC format XML files.
    
    Args:
        directory (str): Directory containing XML annotation files.
        
    Returns:
        dict: Dictionary mapping image paths to annotations.
    """
    result = {}
    
    # Find all XML files in the directory
    for file_name in os.listdir(directory):
        if not file_name.endswith(".xml"):
            continue
        
        file_path = os.path.join(directory, file_name)
        
        try:
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Get image path
            image_path = root.find("path").text if root.find("path") is not None else ""
            if not image_path:
                # Try to construct image path from folder and filename
                folder = root.find("folder").text if root.find("folder") is not None else ""
                filename = root.find("filename").text if root.find("filename") is not None else ""
                image_path = os.path.join(folder, filename)
            
            # Get annotations
            annotations = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                bbox = obj.find("bndbox")
                
                x1 = int(bbox.find("xmin").text)
                y1 = int(bbox.find("ymin").text)
                x2 = int(bbox.find("xmax").text)
                y2 = int(bbox.find("ymax").text)
                
                # Get confidence score if available
                confidence = obj.find("confidence")
                score = float(confidence.text) if confidence is not None else 1.0
                
                annotations.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "label": name
                })
            
            result[image_path] = annotations
        
        except Exception as e:
            print(f"Error parsing XML file {file_path}: {e}")
    
    return result