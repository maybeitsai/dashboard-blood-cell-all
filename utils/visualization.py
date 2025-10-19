"""
Visualization Manager for Blood Cell Classification App
Handles charts, graphs, and data visualization
"""

import logging
import pandas as pd

# Import dependencies with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

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

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationManager:
    """Handles data visualization and charts"""
    
    @staticmethod
    def create_probability_chart(probabilities, class_names=None, class_colors=None):
        """Create probability bar chart"""
        if not PLOTLY_AVAILABLE:
            if STREAMLIT_AVAILABLE:
                st.warning("Plotly not available. Install with: conda install plotly")
            return None
            
        if not CONFIG_AVAILABLE:
            return None
            
        if class_names is None:
            class_names = Config.CLASS_NAMES
        if class_colors is None:
            class_colors = Config.CLASS_COLORS
        
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'Class': class_names,
                'Probability': probabilities * 100,
                'Color': [class_colors[cls] for cls in class_names]
            })
            
            # Sort by probability
            df = df.sort_values('Probability', ascending=True)
            
            # Create bar chart
            fig = px.bar(
                df, 
                x='Probability', 
                y='Class',
                color='Class',
                color_discrete_map=class_colors,
                title="Prediction Confidence by Class",
                labels={'Probability': 'Confidence (%)', 'Class': 'Cell Type'},
                orientation='h'
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=False,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Add percentage labels
            fig.update_traces(
                texttemplate='%{x:.1f}%',
                textposition='outside',
                textfont_size=11
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating probability chart: {str(e)}")
            return None
    
    @staticmethod
    def create_confidence_gauge(confidence):
        """Create confidence gauge chart"""
        if not PLOTLY_AVAILABLE:
            if STREAMLIT_AVAILABLE:
                st.warning("Plotly not available. Install with: conda install plotly")
            return None
            
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level", 'font': {'size': 16}},
                delta={'reference': 80, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 90], 'color': "lightgreen"},
                        {'range': [90, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence gauge: {str(e)}")
            return None
    
    @staticmethod
    def create_results_dataframe(prediction_result):
        """Create detailed results DataFrame"""
        if not CONFIG_AVAILABLE:
            return None
            
        try:
            df = pd.DataFrame({
                'Cell Type': Config.CLASS_NAMES,
                'Probability': [f"{p:.4f}" for p in prediction_result['probabilities']],
                'Percentage': [f"{p*100:.2f}%" for p in prediction_result['probabilities']],
                'Description': [Config.CLASS_DESCRIPTIONS[cls] for cls in Config.CLASS_NAMES]
            })
            
            # Add sorting helper
            df['Prob_Value'] = prediction_result['probabilities']
            df = df.sort_values('Prob_Value', ascending=False)
            df = df.drop('Prob_Value', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating results DataFrame: {str(e)}")
            return None