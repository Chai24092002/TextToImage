import io
from PIL import Image
import gradio as gr
from setup import config





# Function to generate the image
def generate_image(prompt):
    try:
        image_bytes = config({"inputs": prompt})
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        return f"An error occurred: {e}"




# Function to clear the output image
def cancel_image():
    return None  # Return None to clear the image output


# Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 style='color:white; text-align:center;'>Chaithra's Text-To-Image Generator</h1>")

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="e.g., Astronaut riding a horse")
    with gr.Row():
        generate_button = gr.Button("Generate Image")
        cancel_button = gr.Button("Cancel Image")
    with gr.Row():
        output_image = gr.Image(label="Generated Image")

    # Define the action on button click
    generate_button.click(fn=generate_image, inputs=prompt_input, outputs=output_image)
    cancel_button.click(fn=cancel_image, inputs=None, outputs=output_image)  # Cancel button clears the image

# Launch the app
demo.launch()
