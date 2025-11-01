"""
Configuration settings for Blood Cell Classification App
"""

import os

class Config:
    """Application configuration class"""
    
    # Model configuration
    MODEL_PATH = 'models/mobilenetv2_cbam_best.h5'
    IMG_SIZE = (224, 224)
    
    # Class information
    CLASS_NAMES = ['EarlyPreB', 'PreB', 'ProB', 'benign']
    
    CLASS_DESCRIPTIONS = {
        'EarlyPreB': 'Early Pre-B cell malignancy - Early stage of B-lymphoblast development',
        'PreB': 'Pre-B cell malignancy - Intermediate stage of B-lymphoblast development', 
        'ProB': 'Pro-B cell malignancy - Early progenitor B-lymphoblast',
        'benign': 'Healthy/Benign cells - Normal blood cells'
    }
    
    CLASS_COLORS = {
        'EarlyPreB': '#ff6b6b',
        'PreB': '#4ecdc4', 
        'ProB': '#45b7d1',
        'benign': '#96ceb4'
    }
    
    # UI Configuration
    APP_TITLE = "Blood Cell Classification System"
    APP_ICON = "🩸"
    VERSION = "1.0.0"
    
    # File upload settings
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    MAX_FILE_SIZE = 10  # MB
    
    # Prediction thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.95
    GOOD_CONFIDENCE_THRESHOLD = 0.90
    MODERATE_CONFIDENCE_THRESHOLD = 0.80
    
    # Paths
    SRC_DIR = './src'
    MODELS_DIR = './models'
    RESULTS_DIR = './results'
    
    @classmethod
    def get_model_path(cls):
        """Get full model path"""
        return os.path.join(cls.MODELS_DIR, 'mobilenetv2_cbam_best.h5')
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        paths_to_check = [
            cls.SRC_DIR,
            cls.MODELS_DIR,
            cls.get_model_path()
        ]
        
        missing_paths = []
        for path in paths_to_check:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        return missing_paths