# Qwen2-VL Image Analysis Script

This script uses the Qwen2-VL-7B-Instruct model to analyze images and answer questions about them. It's designed to work on both Mac (with MPS acceleration) and other platforms.

## Requirements

- Python 3.8+
- PyTorch 2.2.0+
- CUDA (for GPU acceleration on non-Mac systems)
- MPS-enabled Mac (for Mac systems)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/TrentConley/local_vlm_mac.git
cd local_vlm_mac
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

4. Set up your environment:
   - Create a `.env` file in the root directory
   - Add your image path:
     ```
     IMAGE_PATH=path/to/your/image.jpg
     ```

## Usage

Run the script:
```bash
python test.py
```

The script will:
1. Load the Qwen2-VL model
2. Process the image specified in your `.env` file
3. Answer the query about the image
4. Print the response

To modify the query, edit the `messages` list in `test.py`.

## Platform-Specific Notes

### Mac with Apple Silicon (M1-M4)
- The script will automatically use MPS (Metal Performance Shaders) for acceleration
- Make sure you have the latest version of PyTorch installed
- I am currently running this on a M4 Macbook Pro with 48GB of RAM. Check your memory usage with `top` or `htop` to make sure you have enough. You'll need around 23 GB of unused RAM, and ~20GB storage free.

### Systems with CUDA
- The script will automatically use CUDA for GPU acceleration
- Make sure to have CUDA and appropriate drivers installed

## Model Information

This script uses the Qwen2-VL-7B-Instruct model, which is capable of:
- Processing images of various resolutions and aspect ratios
- Understanding and answering queries about images
- Supporting multiple languages
- Handling complex visual reasoning tasks

For more information about the model, visit the [Qwen2-VL GitHub repository](https://github.com/QwenLM/Qwen-VL).