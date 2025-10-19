"""
Model Manager for Blood Cell Classification App
Handles model loading, caching, and management
"""

import logging
import os

# Import dependencies with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    load_model = None

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading and caching"""
    
    _model = None
    
    @classmethod
    def load_model(cls):
        """Load and cache the trained model"""
        if not TF_AVAILABLE:
            if STREAMLIT_AVAILABLE:
                st.error("TensorFlow not available. Please install in your conda environment: conda install tensorflow")
            return None
            
        if not CONFIG_AVAILABLE:
            if STREAMLIT_AVAILABLE:
                st.error("Configuration not available. Please ensure config.py exists.")
            return None
            
        try:
            model_path = Config.get_model_path()
            
            if not tf.io.gfile.exists(model_path):
                if STREAMLIT_AVAILABLE:
                    st.error(f"Model file not found: {model_path}")
                return None
            
            # Load model
            model = load_model(model_path, compile=False)
            
            # Compile for prediction
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            cls._model = model
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            if STREAMLIT_AVAILABLE:
                st.error(f"Error loading model: {str(e)}")
            return None
    
    @classmethod
    def get_model(cls):
        """Get the cached model"""
        if cls._model is None:
            cls._model = cls.load_model()
        return cls._model