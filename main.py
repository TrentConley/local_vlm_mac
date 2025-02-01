import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_utils import load_model, run_inference
from PIL import Image
import io

app = FastAPI(title="Qwen2-VL API Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model_data = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and processor on startup."""
    global model_data
    model_data = load_model()

@app.post("/process/")
async def process_image(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Process an image with a query using Qwen2-VL model.
    
    Args:
        image: The image file to process
        query: The query text to process with the image
    
    Returns:
        dict: Contains the model's response
    """
    # Verify image format
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        # Read image content
        image_content = await image.read()
        
        # Verify it's a valid image
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()  # Verify it's a valid image
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_content)
            temp_path = temp_file.name
        
        try:
            # Run inference
            response = run_inference(model_data, temp_path, query)
            return {
                "status": "success",
                "response": response
            }
        except Exception as e:
            print(f"Inference error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        # Clean up the temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
