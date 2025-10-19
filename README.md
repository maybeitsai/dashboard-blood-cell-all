# 🩸 Blood Cell Classification System

A modern, modular Streamlit web application for classifying blood cells using a trained MobileNetV2 + CBAM model to detect Acute Lymphoblastic Leukemia (ALL).

Built with clean architecture principles for maintainability and extensibility.

## 🚀 Features

- **🏗️ Clean Architecture**: Modular design with separated concerns
- **🎨 Modern UI**: Responsive interface built with Streamlit
- **⚡ Real-time Classification**: Upload images and get instant predictions
- **📊 Confidence Analysis**: Detailed confidence scores and interpretations
- **📈 Interactive Visualizations**: Charts and gauges using Plotly
- **🔬 Multiple Cell Types**: Classifies 4 types of blood cells:
  - **EarlyPreB**: Early Pre-B cell malignancy
  - **PreB**: Pre-B cell malignancy
  - **ProB**: Pro-B cell malignancy
  - **Benign**: Healthy blood cells
- **🛠️ Modular Components**: Easy to maintain and extend
- **📦 Package Structure**: Professional Python package organization

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Conda Environment**: Recommended (tf2x-directml or similar)
- **Trained Model**: MobileNetV2 + CBAM model (`mobilenetv2_cbam_best.h5`)
- **Dependencies**: See `requirements.txt` for full list
- **Memory**: Minimum 4GB RAM, 8GB+ recommended

## 🛠️ Installation & Setup

### Method 1: Automatic Deployment (Recommended)

1. **Clone/Navigate to project directory**:

   ```bash
   cd dashboard-blood-cell-all
   ```

2. **Run the deployment script**:

   ```bash
   python scripts/deploy.py
   ```

3. **Follow the prompts** to install requirements and launch the app

### Method 2: Manual Setup

1. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify model file exists**:

   ```
   models/mobilenetv2_cbam_best.h5
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 📁 Clean Project Architecture

```
dashboard-blood-cell-all/
├── app.py                      # 🎯 MAIN APPLICATION (only file in root)
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── 📁 config/                  # Configuration package
│   ├── __init__.py
│   └── settings.py            # Application settings & constants
├── 📁 utils/                   # Utility packages (modular design)
│   ├── __init__.py
│   ├── model_manager.py       # Model loading & caching
│   ├── image_processor.py     # Image preprocessing & validation
│   ├── prediction_manager.py  # ML predictions & analysis
│   ├── visualization.py       # Charts, graphs & visualizations
│   └── ui_components.py       # UI styling & components
├── 📁 assets/                  # Static assets & configuration
│   └── .streamlit/            # Streamlit configuration files
├── 📁 scripts/                 # Utility scripts & tools
│   ├── deploy.py              # Deployment automation
│   └── run_app.bat            # Windows batch launcher
├── 📁 models/                  # Machine learning models
│   └── mobilenetv2_cbam_best.h5    # 🎯 Required trained model
├── 📁 src/                     # Original training modules
│   ├── data_preprocessing.py
│   ├── models.py
│   └── training_utils.py
├── 📁 results/                 # Training results & plots
└── 📁 FGD-2.ipynb            # Training notebook
```

### 🏗️ Architecture Benefits

- ✅ **Single Root File**: Only `app.py` in main directory
- ✅ **Modular Design**: Separated concerns for maintainability
- ✅ **Package Structure**: Professional Python organization
- ✅ **Clean Imports**: Organized dependency management
- ✅ **Scalable**: Easy to add new features or modules

## 🎯 Usage

### Running the Application

1. **Activate Environment**:

   ```bash
   conda activate tf2x-directml
   ```

2. **Start the Application**:

   ```bash
   streamlit run app.py
   ```

   Or use the batch script (Windows):

   ```bash
   scripts/run_app.bat
   ```

3. **Access the Web Interface**:
   - Open your browser to `http://localhost:8501`
   - The application will load with a modern, responsive UI

### Using the Blood Cell Classifier

1. **📤 Upload Image**:

   - Click "Choose an image file" in the sidebar
   - Supported formats: PNG, JPG, JPEG
   - Maximum size: 200MB

2. **🔍 View Analysis**:

   - Automatic analysis upon upload
   - Cell type prediction with confidence score
   - Interactive probability distribution charts
   - Detailed cell type information

3. **📊 Explore Results**:
   - Confidence gauge visualization
   - Probability bar charts for all cell types
   - Educational information about detected cell type
   - Interactive visualizations

## 🔧 Configuration

The app configuration is centralized in `config/settings.py`:

- **Model Settings**: Path, input dimensions (224×224)
- **Class Definitions**: Cell type names and descriptions
- **UI Configuration**: Colors, themes, and styling
- **Upload Settings**: File types, size limits
- **Confidence Thresholds**: Prediction interpretation levels
- **Path Management**: Model and data directories

## �️ Technical Details

- **Frontend**: Streamlit 1.22.0+ with custom CSS styling
- **Backend**: TensorFlow 2.10.0 (DirectML optimized)
- **Model Architecture**: MobileNetV2 + CBAM (Convolutional Block Attention Module)
- **Input Specification**: 224×224×3 RGB images
- **Output Classes**: 4 blood cell types with confidence scores
- **Visualization**: Plotly interactive charts and gauges
- **Environment**: Conda tf2x-directml with GPU acceleration

## 📊 Model Performance

- **Base Architecture**: MobileNetV2 backbone for efficient inference
- **Attention Mechanism**: CBAM modules for improved feature focus
- **Training Dataset**: Blood cell classification dataset
- **Best Model**: `mobilenetv2_cbam_best.h5` (optimized weights)

## 🚨 Important Notes

- **Research Use Only**: This tool is intended for research and educational purposes
- **Not for Diagnosis**: Should not be used for medical diagnosis
- **Consult Professionals**: Always consult healthcare professionals for medical decisions
- **Image Quality**: Use clear, high-quality blood cell images for best results

## 🔒 Security & Privacy

- All image processing is done locally
- No data is sent to external servers
- Uploaded images are processed in memory only

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 🔧 Environment Issues

```bash
# Environment not activated
conda activate tf2x-directml

# Missing Streamlit in environment
conda install streamlit

# TensorFlow import errors
conda install tensorflow-gpu=2.10.0
```

#### 📁 File Path Issues

- **Model not found**: Ensure `models/mobilenetv2_cbam_best.h5` exists
- **Import errors**: Check package structure and `__init__.py` files
- **Path separators**: Use forward slashes or `os.path.join()`

#### 🖥️ Memory & Performance

- **Large images**: Automatic resizing to 224×224
- **GPU not detected**: Check DirectML installation
- **Slow predictions**: Verify model loading and GPU acceleration

#### 🌐 Streamlit Issues

```bash
# Clear cache if needed
streamlit cache clear

# Reset Streamlit config
rm -rf ~/.streamlit/

# Force browser refresh
Ctrl + F5 (Windows) / Cmd + Shift + R (Mac)
```

### 📈 Performance Optimization

- **Hardware**: GPU acceleration via DirectML
- **Memory**: Efficient image processing pipeline
- **Caching**: Model and prediction result caching
- **UI**: Responsive design with lazy loading

## 🛡️ System Requirements

### Minimum Requirements

- **RAM**: 4GB (8GB+ recommended)
- **Storage**: ~2GB for dependencies and model
- **Python**: 3.8 or higher
- **OS**: Windows 10/11, macOS, or Linux

### Recommended Setup

- **GPU**: DirectML compatible for acceleration
- **RAM**: 8GB+ for smoother performance
- **SSD**: For faster model loading
- **Browser**: Chrome, Firefox, Safari, or Edge

## � Best Practices

### Image Quality

- **Resolution**: 224×224 pixels (auto-resized)
- **Format**: PNG or JPEG preferred
- **Quality**: High-resolution, clear cell images
- **Background**: Clean, minimal background noise

### Usage Tips

- **Single Cell Focus**: Best results with single, centered cells
- **Good Lighting**: Well-lit images without shadows
- **Sharp Focus**: Avoid blurry or out-of-focus images
- **File Size**: Under 10MB for optimal upload speed

## 📝 Version History

### v1.0.0 - Initial Release

- ✅ MobileNetV2 + CBAM model integration
- ✅ Modern Streamlit UI with custom styling
- ✅ Modular architecture with clean package structure
- ✅ Interactive visualizations with Plotly
- ✅ Confidence gauges and probability charts
- ✅ DirectML GPU acceleration support
- ✅ Real-time classification with confidence analysis
- ✅ Educational cell type information display
- ✅ Responsive design for multiple screen sizes

## 📄 License

This project is for research and educational purposes. The trained model and code are provided as-is for academic use.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions, issues, or contributions:

- Open an issue in the repository
- Check troubleshooting section first
- Provide detailed error messages and environment info

---

<div align="center">

**🩸 Blood Cell Classification Dashboard**

_Powered by MobileNetV2 + CBAM • Built with Streamlit_

</div>

## 🤝 Contributing

This is a research project for blood cell classification. For improvements or issues:

1. Document the issue clearly
2. Include system information
3. Provide sample images (if applicable)
4. Describe expected vs actual behavior

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your institution's guidelines when using medical imaging data.

---

**⚠️ Disclaimer**: This software is for research and educational purposes only. It is not intended for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
