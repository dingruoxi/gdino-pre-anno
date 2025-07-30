# Annotation Tool with Grounding DINO Pre-annotation

This tool allows you to annotate images using Grounding DINO for automatic pre-annotation and provides functionality for manual label editing.

## Features

- **Pre-annotation with Grounding DINO**: Automatically detect objects in images using the state-of-the-art Grounding DINO model
- **Manual Label Editing**: Edit, add, or remove annotations as needed
- **Interactive Annotation Mode**: Draw and edit bounding boxes directly on the image
- **Export Annotations**: Save annotations in common formats (COCO, PASCAL VOC)
- **User-friendly Interface**: Built with Streamlit for an intuitive experience

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
streamlit run app.py
```

## How It Works

1. Upload images or select a folder containing images
2. Specify the objects you want to detect (e.g., "person", "car", "dog")
3. Run pre-annotation with Grounding DINO
4. Edit annotations using one of two methods:
   - **Form-based editing**: Use input fields to precisely adjust coordinates
   - **Interactive editing**: Toggle "Interactive Annotation Mode" to draw and edit bounding boxes directly on the image
5. Export annotations in your preferred format (COCO or PASCAL VOC)

## License

MIT