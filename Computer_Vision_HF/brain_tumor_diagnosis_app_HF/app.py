import gradio as gr
from transformers import pipeline
from PIL import Image

# Define the image classification function
def classify_image(image):
    try:
        # Convert the Gradio image input (which is a NumPy array) to a PIL image
        image = Image.fromarray(image)
        
        # Create the image classification pipeline
        img_class = pipeline(
            "image-classification", model="AMfeta99/vit-base-oxford-brain-tumor"
        )
        
        # Perform image classification
        results = img_class(image)
        
        # Find the result with the highest score
        max_score_result = max(results, key=lambda x: x['score'])
        
        # Extract the predicted label
        predictions = max_score_result['label']
        
        return predictions
    
    except Exception as e:
        # Handle any errors that occur during classification
        return f"Error: {str(e)}"

# Define the Gradio interface
image = gr.Image()
label = gr.Label(num_top_classes=1)
title = "Brain Tumor X-ray Classification"
description = "Worried about whether your brain scan is normal or not? Upload your x-ray and the algorithm will give you an expert opinion. Check out [the original algorithm](https://huggingface.co/AMfeta99/vit-base-oxford-brain-tumor) that this demo is based off of."
article = "<p style='text-align: center'>Image Classification | Demo Model</p>"
demo = gr.Interface(fn=classify_image, inputs=image, outputs=label, description=description, article=article, title=title)

# Launch the Gradio interface
demo.launch(share=True)