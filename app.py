import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch

# Import streamlit_drawable_canvas with error handling
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception as e:
    st.warning(f"Error importing streamlit_drawable_canvas: {e}")
    # Create a placeholder function for st_canvas
    def st_canvas(*args, **kwargs):
        st.error("Interactive canvas is not available due to compatibility issues.")
        return None
    CANVAS_AVAILABLE = False

from utils.grounding_dino import GroundingDINOPredictor
from utils.annotation_utils import save_annotations, load_annotations
from utils.visualization import draw_boxes_on_image
from utils.editor import AnnotationEditor

# Set page configuration
st.set_page_config(page_title="Annotation Tool", layout="wide")

# Initialize session state variables if they don't exist
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_path' not in st.session_state:
    st.session_state.current_image_path = None
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'editor' not in st.session_state:
    st.session_state.editor = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None
if 'current_label' not in st.session_state:
    st.session_state.current_label = ""
if 'editing_annotation_index' not in st.session_state:
    st.session_state.editing_annotation_index = None

# Function to load images from a directory
def load_images_from_dir(directory):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(directory, file))
    return sorted(image_files)

# Function to display the current image with annotations
def display_current_image():
    if st.session_state.current_image is not None:
        # Get annotations for the current image if they exist
        annotations = st.session_state.annotations.get(st.session_state.current_image_path, [])
        
        # Convert PIL Image to numpy array
        img = np.array(st.session_state.current_image)
        
        # Display the image with annotations
        if annotations:
            img = draw_boxes_on_image(img, annotations)
        
        # Display the image
        st.image(img, caption=f"Image: {os.path.basename(st.session_state.current_image_path)}", use_column_width=True)
        
        # Display annotation information in a table
        if annotations:
            # Create a DataFrame with annotations
            annotation_df = pd.DataFrame([
                {
                    "Index": i,
                    "Label": ann["label"],
                    "Score": ann["score"],
                    "X1": ann["bbox"][0],
                    "Y1": ann["bbox"][1],
                    "X2": ann["bbox"][2],
                    "Y2": ann["bbox"][3]
                } for i, ann in enumerate(annotations)
            ])
            st.dataframe(annotation_df, use_container_width=True)

# Function to navigate to the next image
def next_image():
    if st.session_state.current_index < len(st.session_state.image_files) - 1:
        st.session_state.current_index += 1
        load_current_image()

# Function to navigate to the previous image
def prev_image():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        load_current_image()

# Function to process objects drawn or transformed on the canvas
def process_canvas_objects(objects, img_width, img_height):
    # Get current annotations
    annotations = st.session_state.annotations.get(st.session_state.current_image_path, [])
    
    # Process each object from the canvas
    for obj in objects:
        # Check if this is a rectangle
        if obj.get("type") == "rect":
            # Get the object properties
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            scaleX = obj.get("scaleX", 1)
            scaleY = obj.get("scaleY", 1)
            
            # Calculate actual dimensions after scaling
            actual_width = width * scaleX
            actual_height = height * scaleY
            
            # Calculate bounding box coordinates
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(img_width, int(left + actual_width))
            y2 = min(img_height, int(top + actual_height))
            
            # If we're editing an existing annotation
            if st.session_state.editing_annotation_index is not None:
                # Update the existing annotation
                if 0 <= st.session_state.editing_annotation_index < len(annotations):
                    annotations[st.session_state.editing_annotation_index]["bbox"] = [x1, y1, x2, y2]
            else:
                # This is a new annotation
                if st.session_state.current_label:
                    # Create a new annotation
                    new_annotation = {
                        "bbox": [x1, y1, x2, y2],
                        "score": 1.0,  # Manual annotations get a score of 1.0
                        "label": st.session_state.current_label
                    }
                    annotations.append(new_annotation)
                    
                    # Reset current label after adding
                    st.session_state.current_label = ""
    
    # Update annotations in session state
    st.session_state.annotations[st.session_state.current_image_path] = annotations
    
    # Update the editor
    if st.session_state.editor:
        st.session_state.editor.set_annotations(annotations)

# Function to load the current image based on the index
def load_current_image():
    if st.session_state.image_files and 0 <= st.session_state.current_index < len(st.session_state.image_files):
        image_path = st.session_state.image_files[st.session_state.current_index]
        st.session_state.current_image_path = image_path
        st.session_state.current_image = Image.open(image_path)
        
        # Reset edit mode when loading a new image
        st.session_state.edit_mode = False
        st.session_state.editing_annotation_index = None
        st.session_state.current_label = ""
        
        # Initialize the editor for this image if it doesn't exist
        if st.session_state.editor is None or st.session_state.editor.image_path != image_path:
            st.session_state.editor = AnnotationEditor(image_path, st.session_state.current_image)

# Main application layout
st.title("Image Annotation Tool with Grounding DINO")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Image upload section
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp"])
    
    # Directory selection
    st.subheader("Or Select Directory")
    dir_path = st.text_input("Enter directory path")
    if st.button("Load Directory") and dir_path and os.path.isdir(dir_path):
        st.session_state.image_files = load_images_from_dir(dir_path)
        if st.session_state.image_files:
            st.session_state.current_index = 0
            load_current_image()
            st.success(f"Loaded {len(st.session_state.image_files)} images from directory")
        else:
            st.error("No images found in the specified directory")
    
    # Process uploaded files
    if uploaded_files:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files to temp directory
        st.session_state.image_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.image_files.append(file_path)
        
        if st.session_state.image_files:
            st.session_state.current_index = 0
            load_current_image()
            st.success(f"Loaded {len(uploaded_files)} images")
    
    # Pre-annotation section
    st.subheader("Pre-annotation")
    text_prompt = st.text_input("Enter objects to detect (comma-separated)", "person, car, dog, cat")
    box_threshold = st.slider("Box Threshold", 0.1, 0.9, 0.35, 0.05)
    text_threshold = st.slider("Text Threshold", 0.1, 0.9, 0.25, 0.05)
    
    if st.button("Run Pre-annotation") and st.session_state.current_image is not None:
        with st.spinner("Running Grounding DINO for pre-annotation..."):
            try:
                # Initialize the Grounding DINO predictor
                predictor = GroundingDINOPredictor()
                
                # Run prediction on the current image
                boxes, scores, labels = predictor.predict_image(
                    st.session_state.current_image,
                    text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                # Convert to list of dictionaries for easier handling
                annotations = []
                for box, score, label in zip(boxes, scores, labels):
                    annotations.append({
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "score": float(score),
                        "label": label
                    })
                
                # Store annotations for the current image
                st.session_state.annotations[st.session_state.current_image_path] = annotations
                
                # Update the editor with new annotations
                if st.session_state.editor:
                    st.session_state.editor.set_annotations(annotations)
                
                st.success(f"Found {len(annotations)} objects")
            except Exception as e:
                st.error(f"Error during pre-annotation: {str(e)}")
    
    # Export annotations
    st.subheader("Export Annotations")
    export_format = st.selectbox("Export Format", ["COCO", "PASCAL VOC"])
    export_path = st.text_input("Export Directory")
    
    if st.button("Export Annotations") and export_path:
        try:
            os.makedirs(export_path, exist_ok=True)
            save_annotations(st.session_state.annotations, export_format, export_path)
            st.success(f"Annotations exported to {export_path}")
        except Exception as e:
            st.error(f"Error exporting annotations: {str(e)}")
    
    # Navigation buttons
    st.subheader("Navigation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            prev_image()
    with col2:
        if st.button("Next"):
            next_image()

# Main content area
if st.session_state.current_image is None:
    st.info("Please upload images or select a directory to start annotation")
else:
    # Display the current image with annotations
    
    # Display the current image with annotations
    display_current_image()
    
    # Manual annotation editing section
    st.subheader("Manual Annotation Editing")
    
    if st.session_state.edit_mode:
        # Interactive annotation editing UI
        st.info("Interactive annotation mode enabled. Draw rectangles on the image above to create annotations.")
        
        # Label input for new annotations
        if st.session_state.editing_annotation_index is None:
            st.session_state.current_label = st.text_input("Label for new annotation", st.session_state.current_label)
            st.caption("Draw a rectangle on the image to create an annotation with this label")
        
        # Annotation selection for editing
        annotations = st.session_state.annotations.get(st.session_state.current_image_path, [])
        if annotations:
            st.subheader("Edit Existing Annotations")
            
            # Select annotation to edit
            selected_idx = st.selectbox(
                "Select Annotation to Edit", 
                range(len(annotations)),
                format_func=lambda i: f"{annotations[i]['label']} (Score: {annotations[i]['score']:.2f})",
                index=0 if st.session_state.editing_annotation_index is None else st.session_state.editing_annotation_index
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Edit Selected"):
                    st.session_state.editing_annotation_index = selected_idx
                    # Convert annotation to canvas object
                    st.experimental_rerun()
            
            with col2:
                if st.button("Delete Selected"):
                    annotations.pop(selected_idx)
                    st.session_state.annotations[st.session_state.current_image_path] = annotations
                    if st.session_state.editor:
                        st.session_state.editor.set_annotations(annotations)
                    st.session_state.editing_annotation_index = None
                    st.experimental_rerun()
            
            with col3:
                if st.session_state.editing_annotation_index is not None and st.button("Finish Editing"):
                    st.session_state.editing_annotation_index = None
                    st.experimental_rerun()
    else:
        # Traditional form-based annotation editing
        
        # Add new annotation
        with st.expander("Add New Annotation"):
            new_label = st.text_input("Label")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("X1", 0, st.session_state.current_image.width, 0)
                y1 = st.number_input("Y1", 0, st.session_state.current_image.height, 0)
            with col2:
                x2 = st.number_input("X2", 0, st.session_state.current_image.width, 100)
                y2 = st.number_input("Y2", 0, st.session_state.current_image.height, 100)
            
            if st.button("Add Annotation") and new_label:
                new_annotation = {
                    "bbox": [x1, y1, x2, y2],
                    "score": 1.0,  # Manual annotations get a score of 1.0
                    "label": new_label
                }
                
                # Get current annotations or initialize empty list
                current_annotations = st.session_state.annotations.get(st.session_state.current_image_path, [])
                current_annotations.append(new_annotation)
                
                # Update annotations
                st.session_state.annotations[st.session_state.current_image_path] = current_annotations
                
                # Update the editor
                if st.session_state.editor:
                    st.session_state.editor.set_annotations(current_annotations)
                
                st.success(f"Added annotation for {new_label}")
                st.experimental_rerun()
        
        # Edit or delete existing annotations
        with st.expander("Edit/Delete Annotations"):
            annotations = st.session_state.annotations.get(st.session_state.current_image_path, [])
            if not annotations:
                st.info("No annotations to edit")
            else:
                selected_idx = st.selectbox("Select Annotation", range(len(annotations)), 
                                          format_func=lambda i: f"{annotations[i]['label']} (Score: {annotations[i]['score']:.2f})")
                
                selected_annotation = annotations[selected_idx]
                
                # Edit form
                edit_label = st.text_input("Edit Label", selected_annotation["label"])
                col1, col2 = st.columns(2)
                with col1:
                    edit_x1 = st.number_input("Edit X1", 0, st.session_state.current_image.width, int(selected_annotation["bbox"][0]))
                    edit_y1 = st.number_input("Edit Y1", 0, st.session_state.current_image.height, int(selected_annotation["bbox"][1]))
                with col2:
                    edit_x2 = st.number_input("Edit X2", 0, st.session_state.current_image.width, int(selected_annotation["bbox"][2]))
                    edit_y2 = st.number_input("Edit Y2", 0, st.session_state.current_image.height, int(selected_annotation["bbox"][3]))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update Annotation"):
                        annotations[selected_idx] = {
                            "bbox": [edit_x1, edit_y1, edit_x2, edit_y2],
                            "score": selected_annotation["score"],
                            "label": edit_label
                        }
                        
                        # Update annotations
                        st.session_state.annotations[st.session_state.current_image_path] = annotations
                        
                        # Update the editor
                        if st.session_state.editor:
                            st.session_state.editor.set_annotations(annotations)
                        
                        st.success("Annotation updated")
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Delete Annotation"):
                        annotations.pop(selected_idx)
                        
                        # Update annotations
                        st.session_state.annotations[st.session_state.current_image_path] = annotations
                        
                        # Update the editor
                        if st.session_state.editor:
                            st.session_state.editor.set_annotations(annotations)
                        
                        st.success("Annotation deleted")
                        st.experimental_rerun()

# Display image count and current position
if st.session_state.image_files:
    st.caption(f"Image {st.session_state.current_index + 1} of {len(st.session_state.image_files)}")