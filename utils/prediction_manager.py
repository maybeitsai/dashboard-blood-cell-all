"""
Prediction Manager for Blood Cell Classification App
Handles model predictions and confidence analysis
"""

import logging
import numpy as np

# Import dependencies with fallbacks
try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None

# Configure logging
logger = logging.getLogger(__name__)

class PredictionManager:
    """Handles model predictions and analysis"""
    
    @staticmethod
    def predict(model, processed_image):
        """Make prediction on processed image"""
        if not CONFIG_AVAILABLE:
            return None
            
        try:
            # Get prediction
            predictions = model.predict(processed_image, verbose=0)
            
            # Extract probabilities
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = Config.CLASS_NAMES[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            return {
                'probabilities': probabilities,
                'predicted_class_idx': predicted_class_idx,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': dict(zip(Config.CLASS_NAMES, probabilities))
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    @staticmethod
    def interpret_confidence(confidence):
        """Interpret confidence level"""
        if not CONFIG_AVAILABLE:
            return "unknown", "Configuration not available"
            
        if confidence >= Config.HIGH_CONFIDENCE_THRESHOLD:
            return "very_high", "🎯 Very High Confidence - Strong prediction"
        elif confidence >= Config.GOOD_CONFIDENCE_THRESHOLD:
            return "good", "✅ Good Confidence - Reliable prediction"
        elif confidence >= Config.MODERATE_CONFIDENCE_THRESHOLD:
            return "moderate", "⚠️ Moderate Confidence - Consider additional analysis"
        else:
            return "low", "❌ Low Confidence - Uncertain prediction"