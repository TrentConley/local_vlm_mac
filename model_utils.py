import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
from PIL import Image

def get_device():
    """Get the appropriate device for inference."""
    if torch.backends.mps.is_available():
        return "mps"  # For Mac M1/M2
    elif torch.cuda.is_available():
        return "cuda"  # For NVIDIA GPUs
    return "cpu"  # Fallback

def load_model():
    """Load the Qwen2-VL model and processor."""
    device = get_device()
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True  # Important for loading Qwen models
    )
    
    return model, processor

def run_inference(model_data, image_path, query):
    """Run inference on an image with a query."""
    try:
        model, processor = model_data
        
        # Open and verify the image
        image = Image.open(image_path)
        
        # Prepare the messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        # Process the input
        inputs = processor.chat_process(messages, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the response
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        return response
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise Exception(f"Failed to process image: {str(e)}") 