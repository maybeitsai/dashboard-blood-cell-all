"""
🩸 Blood Cell Classification System
===================================

A modern Streamlit web application for classifying blood cells using a trained 
MobileNetV2 + CBAM model to detect Acute Lymphoblastic Leukemia (ALL).

Author: Blood Cell Classification Team
Version: 1.0.0
License: MIT
"""

import streamlit as st
import sys
import os
from PIL import Image

# Add project directories to path
sys.path.append('./config')
sys.path.append('./utils')

# Import application modules
from config import Config
from utils import (
    ModelManager, 
    ImageProcessor, 
    PredictionManager,
    VisualizationManager, 
    UIComponents
)

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar():
    """Create application sidebar with model information and instructions"""
    with st.sidebar:
        st.markdown("### 🔧 Model Information")
        
        st.markdown("""
        <div class="sidebar-content">
            <p><span class="status-indicator status-success"></span><strong>Model:</strong> MobileNetV2 + CBAM</p>
            <p><span class="status-indicator status-info"></span><strong>Input Size:</strong> 224×224 pixels</p>
            <p><span class="status-indicator status-success"></span><strong>Classes:</strong> 4 cell types</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Cell Type Information")
        for cls, desc in Config.CLASS_DESCRIPTIONS.items():
            color = Config.CLASS_COLORS[cls]
            st.markdown(f"""
            <div class="sidebar-content">
                <p style="color: {color}; font-weight: bold;">● {cls}</p>
                <p style="font-size: 0.9em; margin-top: -10px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        - This tool is for research purposes only
        - Not intended for medical diagnosis
        - Always consult healthcare professionals
        - Upload clear, high-quality cell images
        """)

def create_upload_section():
    """Create file upload section"""
    st.markdown("### 📁 Upload Blood Cell Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=Config.ALLOWED_EXTENSIONS,
        help="Upload a clear image of blood cells for classification"
    )
    
    return uploaded_file

def display_image_info(image, uploaded_file):
    """Display uploaded image and its information"""
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get image info
    img_info = ImageProcessor.get_image_info(image)
    
    st.markdown(f"""
    <div class="info-card">
        <strong>Image Information:</strong><br>
        📏 Size: {img_info.get('size', ['Unknown', 'Unknown'])[0]} × {img_info.get('size', ['Unknown', 'Unknown'])[1]} pixels<br>
        📊 Mode: {img_info.get('mode', 'Unknown')}<br>
        📁 File: {uploaded_file.name}
    </div>
    """, unsafe_allow_html=True)

def display_prediction_results(prediction_result):
    """Display prediction results with styling"""
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    color = Config.CLASS_COLORS[predicted_class]
    description = Config.CLASS_DESCRIPTIONS[predicted_class]
    
    # Main prediction display
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-result" style="color: {color};">
            {predicted_class}
        </div>
        <div class="confidence-score">
            Confidence: {confidence:.2%}
        </div>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence interpretation
    confidence_level, message = PredictionManager.interpret_confidence(confidence)
    
    if confidence_level == "very_high":
        st.success(message)
    elif confidence_level == "good":
        st.info(message)
    elif confidence_level == "moderate":
        st.warning(message)
    else:
        st.error(message)

def create_detailed_analysis(prediction_result):
    """Create detailed analysis section with charts and tables"""
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Probability chart
        chart = VisualizationManager.create_probability_chart(
            prediction_result['probabilities']
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        # Confidence gauge
        gauge = VisualizationManager.create_confidence_gauge(
            prediction_result['confidence']
        )
        if gauge:
            st.plotly_chart(gauge, use_container_width=True)
    
    # Results table
    st.markdown("### 📋 All Class Probabilities")
    results_df = VisualizationManager.create_results_dataframe(prediction_result)
    if results_df is not None:
        st.dataframe(results_df, use_container_width=True)

def main():
    """Main application function"""
    
    # Load custom CSS and create header
    UIComponents.load_custom_css()
    UIComponents.create_header()
    
    # Validate paths
    missing_paths = Config.validate_paths()
    if missing_paths:
        st.error(f"Missing required files/directories: {', '.join(missing_paths)}")
        st.info("Please ensure your trained model is in the models/ directory")
        st.stop()
    
    # Load model
    model = ModelManager.get_model()
    if model is None:
        st.error("Failed to load the model. Please check the model file and dependencies.")
        st.stop()
    
    # Create sidebar
    create_sidebar()
    
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload section
        uploaded_file = create_upload_section()
        
        if uploaded_file is not None:
            # Validate uploaded file
            is_valid, message = ImageProcessor.validate_image(uploaded_file)
            
            if not is_valid:
                st.error(f"Invalid file: {message}")
            else:
                # Display image and info
                image = Image.open(uploaded_file)
                display_image_info(image, uploaded_file)
    
    with col2:
        st.markdown("### 🤖 Prediction Results")
        
        if uploaded_file is not None and 'image' in locals():
            with st.spinner("🔄 Analyzing image..."):
                # Preprocess image
                processed_image, display_image = ImageProcessor.preprocess_image(image)
                
                if processed_image is not None:
                    # Make prediction
                    prediction_result = PredictionManager.predict(model, processed_image)
                    
                    if prediction_result is not None:
                        # Display results
                        display_prediction_results(prediction_result)
                        
                        # Store prediction for detailed analysis
                        st.session_state.prediction_result = prediction_result
                    else:
                        st.error("Failed to make prediction on the image.")
                else:
                    st.error("Failed to process the uploaded image.")
        else:
            st.markdown("""
            <div class="upload-area">
                <h3>👆 Please upload an image above</h3>
                <p>Supported formats: PNG, JPG, JPEG, BMP, TIFF</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis section (only if prediction exists)
    if hasattr(st.session_state, 'prediction_result') and uploaded_file is not None:
        create_detailed_analysis(st.session_state.prediction_result)
    
    # Footer
    UIComponents.create_footer()

if __name__ == "__main__":
    main()