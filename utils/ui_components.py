"""
UI Components for Blood Cell Classification App
Handles custom styling, headers, footers, and UI elements
"""

# Import dependencies with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None

class UIComponents:
    """Custom UI components and styling"""
    
    @staticmethod
    def load_custom_css():
        """Load custom CSS styling"""
        if not STREAMLIT_AVAILABLE:
            return
            
        st.markdown("""
        <style>
        /* Main container styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Cards styling */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .prediction-result {
            font-size: 2rem;
            font-weight: bold;
            margin: 1rem 0;
            color: #2c3e50;
        }
        
        .confidence-score {
            font-size: 1.5rem;
            margin: 0.5rem 0;
            color: #34495e;
        }
        
        /* Sidebar styling */
        .sidebar-content {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-success { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .status-info { background-color: #17a2b8; }
        
        /* Upload area */
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: #f8f9ff;
            margin: 1rem 0;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 { font-size: 2rem; }
            .prediction-result { font-size: 1.5rem; }
            .confidence-score { font-size: 1.2rem; }
        }
        
        /* Hide streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_header():
        """Create main application header"""
        if not STREAMLIT_AVAILABLE or not CONFIG_AVAILABLE:
            return
            
        st.markdown(f"""
        <div class="main-header">
            <h1>{Config.APP_ICON} {Config.APP_TITLE}</h1>
            <p>AI-Powered Detection for Acute Lymphoblastic Leukemia (ALL) using MobileNetV2 + CBAM</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_footer():
        """Create application footer"""
        if not STREAMLIT_AVAILABLE or not CONFIG_AVAILABLE:
            return
            
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; color: #666;">
            <p>🔬 {Config.APP_TITLE} v{Config.VERSION} | 
            Built with Streamlit & TensorFlow | 
            Powered by MobileNetV2 + CBAM Architecture</p>
            <p><small>⚠️ For research and educational purposes only - Not for medical diagnosis</small></p>
        </div>
        """, unsafe_allow_html=True)