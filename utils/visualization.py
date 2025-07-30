import cv2
import numpy as np
import random

# Generate a list of distinct colors for visualization
def generate_colors(n):
    """
    Generate n distinct colors for visualization.
    
    Args:
        n (int): Number of colors to generate.
        
    Returns:
        list: List of (B, G, R) color tuples.
    """
    colors = []
    for i in range(n):
        # Use HSV color space to generate distinct colors
        hue = i / n
        saturation = 0.9
        value = 0.9
        
        # Convert HSV to RGB
        h = hue * 360
        s = saturation
        v = value
        
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append((b, g, r))  # OpenCV uses BGR
    
    return colors

# Cache for label colors
label_colors = {}

def get_color_for_label(label):
    """
    Get a consistent color for a label.
    
    Args:
        label (str): The label to get a color for.
        
    Returns:
        tuple: (B, G, R) color tuple.
    """
    global label_colors
    
    if label not in label_colors:
        # Generate a random color for this label
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        label_colors[label] = color
    
    return label_colors[label]

def draw_boxes_on_image(image, annotations, thickness=2, font_scale=0.6):
    """
    Draw bounding boxes and labels on an image.
    
    Args:
        image (numpy.ndarray): The image to draw on.
        annotations (list): List of annotation dictionaries with 'bbox', 'label', and 'score'.
        thickness (int): Line thickness for bounding boxes.
        font_scale (float): Font scale for labels.
        
    Returns:
        numpy.ndarray: Image with bounding boxes and labels drawn.
    """
    # Make a copy of the image to avoid modifying the original
    img_with_boxes = image.copy()
    
    for ann in annotations:
        # Get bounding box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in ann["bbox"]]
        
        # Get label and score
        label = ann["label"]
        score = ann.get("score", 1.0)
        
        # Get color for this label
        color = get_color_for_label(label)
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_text = f"{label}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            img_with_boxes,
            label_text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
    
    return img_with_boxes