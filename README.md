# Qwen2-VL FastAPI Server

This is a FastAPI server that uses the Qwen2-VL-7B-Instruct model for processing images with queries. The server is designed to work on both Mac (with MPS acceleration) and other platforms like AWS (with CUDA).

## Requirements

- Python 3.8+
- PyTorch 2.2.0+
- CUDA (for GPU acceleration on non-Mac systems)
- MPS-enabled Mac (for Mac systems)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

2. API Endpoints:

- POST `/process/`
  - Parameters:
    - `image`: Image file (multipart/form-data)
    - `query`: Text query about the image
  - Returns: JSON with model's response

3. Example curl request:
```bash
curl -X POST "http://localhost:8000/process/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg" \
  -F "query=Describe this image"
```

## Platform-Specific Notes

### Mac with Apple Silicon (M1/M2)
- The server will automatically use MPS (Metal Performance Shaders) for acceleration
- Make sure you have the latest version of PyTorch installed

### AWS P3 Instance
- The server will automatically use CUDA for GPU acceleration
- Make sure to use an AMI with CUDA and appropriate drivers installed
- Recommended instance types: p3.2xlarge or better

## Model Information

This server uses the Qwen2-VL-7B-Instruct model, which is capable of:
- Processing images of various resolutions and aspect ratios
- Understanding and answering queries about images
- Supporting multiple languages
- Handling complex visual reasoning tasks

For more information about the model, visit the [Qwen2-VL GitHub repository](https://github.com/QwenLM/Qwen-VL). 