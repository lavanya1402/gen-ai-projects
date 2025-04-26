import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Optional: Disable the symlink warning
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the pretrained processor and model with use_fast=True for speed optimization
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image (replace with actual path)
img_path = r"C:\Users\lavan\Downloads\view-wild-lion-nature.jpg"

# Convert it into an RGB format
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"

# Process the image and generate inputs
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print(caption)
