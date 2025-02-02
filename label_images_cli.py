#!/usr/bin/env python3

import os
import glob
import json
from datetime import datetime
import logging
import subprocess
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('labeling.log')
    ]
)

logger = logging.getLogger(__name__)

class ImageLabeler:
    def __init__(self, image_dir="images"):
        self.image_dir = image_dir
        self.labels = {}
        self.current_index = 0
        
        # Ensure image directory exists
        if not os.path.exists(image_dir):
            logger.info(f"Creating image directory: {image_dir}")
            os.makedirs(image_dir)
        
        # Get list of images
        self.image_files = sorted(
            glob.glob(os.path.join(image_dir, "*.png")) + 
            glob.glob(os.path.join(image_dir, "*.jpg")) + 
            glob.glob(os.path.join(image_dir, "*.jpeg"))
        )
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        # Load existing labels if any
        self.load_existing_labels()

    def load_existing_labels(self):
        """Load existing labels from JSON file if it exists."""
        label_file = "ground_truth_labels.json"
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    self.labels = json.load(f)
                logger.info(f"Loaded {len(self.labels)} existing labels")
            except Exception as e:
                logger.error(f"Error loading labels: {str(e)}")
                self.labels = {}

    def save_labels(self):
        """Save labels to both current and timestamped JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ground_truth_labels_{timestamp}.json"
        
        try:
            # Save both versions
            with open("ground_truth_labels.json", 'w') as f:
                json.dump(self.labels, f, indent=2)
            with open(filename, 'w') as f:
                json.dump(self.labels, f, indent=2)
            logger.info(f"Labels saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving labels: {str(e)}")
            return False

    def show_image(self, image_path):
        """Open image with system default viewer."""
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(image_path)
            
            if os.name == 'nt':  # Windows
                os.startfile(abs_path)
            elif os.name == 'posix':  # macOS and Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(['open', abs_path])
                else:  # Linux
                    subprocess.run(['xdg-open', abs_path])
            
            return True
        except Exception as e:
            logger.error(f"Error showing image: {str(e)}")
            return False

    def run(self):
        """Main labeling loop."""
        print("\nImage Labeling Tool")
        print("------------------")
        print("Press Ctrl+C to save and quit at any time")
        print("Enter 'q' to save and quit")
        print("Enter 's' to save current progress")
        print("Just press Enter to skip an image\n")

        try:
            while self.current_index < len(self.image_files):
                image_path = self.image_files[self.current_index]
                print(f"\nImage {self.current_index + 1} of {len(self.image_files)}")
                print(f"File: {os.path.basename(image_path)}")
                
                # Show existing label if it exists
                if image_path in self.labels:
                    print(f"Current label: {self.labels[image_path]}")
                
                # Show the image
                if not self.show_image(image_path):
                    print("Error showing image. Skipping...")
                    self.current_index += 1
                    continue
                
                # Get input
                value = input("Enter fuel quantity (or q to quit, s to save): ").strip().lower()
                
                if value == 'q':
                    break
                elif value == 's':
                    self.save_labels()
                    continue
                elif value == '':
                    print("Skipping image...")
                elif value.replace('.', '').isdigit():  # Allow decimal points
                    self.labels[image_path] = float(value)
                    print(f"Saved: {value}")
                else:
                    print("Invalid input. Please enter a number or command.")
                    continue
                
                self.current_index += 1
            
            # Save final results
            self.save_labels()
            print("\nLabeling complete!")
            print(f"Processed {len(self.labels)} images")
            
        except KeyboardInterrupt:
            print("\nSaving and exiting...")
            self.save_labels()
        except Exception as e:
            logger.error(f"Error during labeling: {str(e)}")
            self.save_labels()
            raise

def main():
    try:
        labeler = ImageLabeler()
        labeler.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 