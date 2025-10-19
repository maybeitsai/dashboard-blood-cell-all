"""
Image Processor for Blood Cell Classification App
Handles image preprocessing, validation, and analysis
"""

import logging
import numpy as np
from PIL import Image

# Import dependencies with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    try:
        # Try importing cv2 again in case it's installed as opencv-python-headless
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image preprocessing and analysis"""
    
    @staticmethod
    def validate_image(image_file):
        """Validate uploaded image file"""
        if not CONFIG_AVAILABLE:
            return False, "Configuration not available"
            
        try:
            # Check file size
            if image_file.size > Config.MAX_FILE_SIZE * 1024 * 1024:
                return False, f"File size too large. Maximum {Config.MAX_FILE_SIZE}MB allowed."
            
            # Check file extension
            file_extension = image_file.name.split('.')[-1].lower()
            if file_extension not in Config.ALLOWED_EXTENSIONS:
                return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            
            # Try to open image
            image = Image.open(image_file)
            
            # Check image mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                return False, "Invalid image mode. Please use RGB, RGBA, or grayscale images."
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    @staticmethod
    def preprocess_image(image, target_size=None):
        """Preprocess image for model prediction"""
        if not CV2_AVAILABLE:
            # Fallback to PIL/numpy preprocessing when OpenCV is not available
            return ImageProcessor._preprocess_image_fallback(image, target_size)
            
        if not CONFIG_AVAILABLE:
            return None, None
            
        try:
            if target_size is None:
                target_size = Config.IMG_SIZE
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Handle different image modes
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif img_array.shape[2] == 1:  # Grayscale with channel
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Resize image
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch, img_resized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None, None
    
    @staticmethod
    def _preprocess_image_fallback(image, target_size=None):
        """Fallback preprocessing using PIL when OpenCV is not available"""
        if not CONFIG_AVAILABLE:
            return None, None
            
        try:
            if target_size is None:
                target_size = Config.IMG_SIZE
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image using PIL
            img_resized = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img_resized)
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch, img_array
            
        except Exception as e:
            logger.error(f"Error in fallback preprocessing: {str(e)}")
            if STREAMLIT_AVAILABLE:
                st.error(f"Failed to process image: {str(e)}")
            return None, None
    
    @staticmethod
    def get_image_info(image):
        """Get detailed image information"""
        try:
            info = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': 'transparency' in image.info
            }
            return info
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            return {}