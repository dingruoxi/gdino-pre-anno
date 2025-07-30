import numpy as np
from PIL import Image

class AnnotationEditor:
    def __init__(self, image_path, image):
        """
        Initialize the annotation editor.
        
        Args:
            image_path (str): Path to the image file.
            image (PIL.Image): The image object.
        """
        self.image_path = image_path
        self.image = image
        self.annotations = []
        self.selected_annotation_index = None
        
        # Image dimensions
        self.width, self.height = image.size
    
    def set_annotations(self, annotations):
        """
        Set the annotations for the current image.
        
        Args:
            annotations (list): List of annotation dictionaries.
        """
        self.annotations = annotations
        self.selected_annotation_index = None
    
    def add_annotation(self, bbox, label, score=1.0):
        """
        Add a new annotation.
        
        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            label (str): Object label.
            score (float): Confidence score.
            
        Returns:
            int: Index of the new annotation.
        """
        annotation = {
            "bbox": bbox,
            "label": label,
            "score": score
        }
        
        self.annotations.append(annotation)
        return len(self.annotations) - 1
    
    def update_annotation(self, index, bbox=None, label=None, score=None):
        """
        Update an existing annotation.
        
        Args:
            index (int): Index of the annotation to update.
            bbox (list, optional): New bounding box coordinates.
            label (str, optional): New object label.
            score (float, optional): New confidence score.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if index < 0 or index >= len(self.annotations):
            return False
        
        if bbox is not None:
            self.annotations[index]["bbox"] = bbox
        
        if label is not None:
            self.annotations[index]["label"] = label
        
        if score is not None:
            self.annotations[index]["score"] = score
        
        return True
    
    def delete_annotation(self, index):
        """
        Delete an annotation.
        
        Args:
            index (int): Index of the annotation to delete.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if index < 0 or index >= len(self.annotations):
            return False
        
        self.annotations.pop(index)
        
        # Reset selected annotation if it was deleted
        if self.selected_annotation_index == index:
            self.selected_annotation_index = None
        elif self.selected_annotation_index > index:
            self.selected_annotation_index -= 1
        
        return True
    
    def select_annotation(self, x, y):
        """
        Select an annotation based on a point (x, y).
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            
        Returns:
            int: Index of the selected annotation, or None if no annotation was selected.
        """
        selected_index = None
        min_area = float('inf')
        
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann["bbox"]
            
            # Check if point is inside the bounding box
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Calculate area of the bounding box
                area = (x2 - x1) * (y2 - y1)
                
                # Select the smallest bounding box containing the point
                if area < min_area:
                    min_area = area
                    selected_index = i
        
        self.selected_annotation_index = selected_index
        return selected_index
    
    def get_selected_annotation(self):
        """
        Get the currently selected annotation.
        
        Returns:
            dict: The selected annotation, or None if no annotation is selected.
        """
        if self.selected_annotation_index is None:
            return None
        
        return self.annotations[self.selected_annotation_index]
    
    def move_annotation(self, index, dx, dy):
        """
        Move an annotation by the specified delta.
        
        Args:
            index (int): Index of the annotation to move.
            dx (int): Change in x coordinate.
            dy (int): Change in y coordinate.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if index < 0 or index >= len(self.annotations):
            return False
        
        x1, y1, x2, y2 = self.annotations[index]["bbox"]
        
        # Apply deltas
        x1 += dx
        y1 += dy
        x2 += dx
        y2 += dy
        
        # Ensure the box stays within the image boundaries
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        # Update the annotation
        self.annotations[index]["bbox"] = [x1, y1, x2, y2]
        
        return True
    
    def resize_annotation(self, index, edge, dx, dy):
        """
        Resize an annotation by moving one of its edges.
        
        Args:
            index (int): Index of the annotation to resize.
            edge (str): Which edge to move ('top', 'bottom', 'left', 'right').
            dx (int): Change in x coordinate.
            dy (int): Change in y coordinate.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if index < 0 or index >= len(self.annotations):
            return False
        
        x1, y1, x2, y2 = self.annotations[index]["bbox"]
        
        # Apply changes based on which edge is being moved
        if edge == "top":
            y1 += dy
        elif edge == "bottom":
            y2 += dy
        elif edge == "left":
            x1 += dx
        elif edge == "right":
            x2 += dx
        else:
            return False
        
        # Ensure the box stays within the image boundaries and has positive dimensions
        x1 = max(0, min(x1, x2 - 1, self.width - 1))
        y1 = max(0, min(y1, y2 - 1, self.height - 1))
        x2 = max(x1 + 1, min(x2, self.width - 1))
        y2 = max(y1 + 1, min(y2, self.height - 1))
        
        # Update the annotation
        self.annotations[index]["bbox"] = [x1, y1, x2, y2]
        
        return True
    
    def get_annotations(self):
        """
        Get all annotations for the current image.
        
        Returns:
            list: List of annotation dictionaries.
        """
        return self.annotations