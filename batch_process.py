from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from dotenv import load_dotenv
import os
import glob
import json
from datetime import datetime

def setup_model_and_processor():
    """Initialize the model and processor."""
    # Load model with float16 for better memory efficiency
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Use the processor with reduced token count
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )

    # Check if MPS is available, otherwise use CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    return model, processor, device

def process_single_image(image_path, model, processor, device):
    """Process a single image and return the result."""
    try:
        # Load and process image
        image = Image.open(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "How many gallons of fuel was consumed?"},
                ],
            }
        ]

        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {
            "image_path": image_path,
            "result": output_text[0],
            "status": "success"
        }
    except Exception as e:
        return {
            "image_path": image_path,
            "result": None,
            "status": "error",
            "error_message": str(e)
        }

def main():
    # Load environment variables
    load_dotenv()

    # Setup model and processor
    print("Setting up model and processor...")
    model, processor, device = setup_model_and_processor()

    # Get all image files
    image_files = glob.glob("images/*.png") + glob.glob("images/*.jpeg")
    print(f"Found {len(image_files)} images to process")

    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_path}")
        result = process_single_image(image_path, model, processor, device)
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main() 