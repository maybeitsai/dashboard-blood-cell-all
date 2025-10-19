"""
Visualization Manager for Blood Cell Classification App
Handles charts, graphs, and data visualization
"""

import logging
import pandas as pd
import numpy as np

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
            if STREAMLIT_AVAILABLE:
                st.error("Configuration not available")
            return None
            
        if class_names is None:
            class_names = Config.CLASS_NAMES
        if class_colors is None:
            class_colors = Config.CLASS_COLORS
        
        try:
            # Ensure probabilities is a numpy array and flatten it
            if hasattr(probabilities, 'numpy'):
                probabilities = probabilities.numpy()
            probabilities = np.array(probabilities).flatten()
            
            # Validate data
            if len(probabilities) != len(class_names):
                if STREAMLIT_AVAILABLE:
                    st.error(f"Mismatch: {len(probabilities)} probabilities vs {len(class_names)} classes")
                return None
            
            # Convert to percentages
            percentages = probabilities * 100
            
            # Create DataFrame with explicit data types
            data = {
                'Class': class_names,
                'Probability': percentages.tolist(),
                'Raw_Prob': probabilities.tolist()
            }
            
            df = pd.DataFrame(data)
            
            # Sort by probability for better visualization
            df = df.sort_values('Probability', ascending=True)
            
            # Create colors list in the same order as sorted dataframe
            colors = [class_colors.get(cls, '#1f77b4') for cls in df['Class']]
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Add bars with better formatting
            fig.add_trace(go.Bar(
                x=df['Probability'],
                y=df['Class'],
                orientation='h',
                marker=dict(color=colors),
                text=[f'{p:.2f}%' if p >= 0.01 else f'{raw*100:.4f}%' 
                      for p, raw in zip(df['Probability'], df['Raw_Prob'])],
                textposition='outside',
                textfont=dict(size=12),
                name='Confidence',
                hovertemplate='<b>%{y}</b><br>' +
                             'Confidence: %{x:.2f}%<br>' +
                             'Raw Probability: %{customdata:.6f}<extra></extra>',
                customdata=df['Raw_Prob']
            ))
            
            # Calculate appropriate x-axis range
            max_prob = max(percentages)
            x_range_max = max(max_prob * 1.2, 10)  # At least 10% for small values
            
            # Update layout with better spacing and formatting
            fig.update_layout(
                title=dict(
                    text="Prediction Confidence by Class",
                    x=0.5,
                    font=dict(size=18, color='#2c3e50')
                ),
                xaxis=dict(
                    title="Confidence (%)",
                    range=[0, x_range_max],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='.2f'
                ),
                yaxis=dict(
                    title="Cell Type",
                    tickfont=dict(size=12)
                ),
                height=450,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=120, r=80, t=60, b=50)
            )
            
            # Add a subtle background
            fig.update_layout(
                shapes=[
                    dict(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=0, y0=0, x1=1, y1=1,
                        fillcolor="rgba(248,249,250,0.8)",
                        layer="below",
                        line_width=0,
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating probability chart: {str(e)}")
            if STREAMLIT_AVAILABLE:
                st.error(f"Chart creation error: {str(e)}")
                # Create a simple fallback chart
                return VisualizationManager.create_simple_fallback_chart(probabilities, class_names, class_colors)
            return None
    
    @staticmethod
    def create_simple_fallback_chart(probabilities, class_names, class_colors):
        """Create a simple fallback chart when main chart fails"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
                
            # Ensure probabilities is a simple list
            if hasattr(probabilities, 'numpy'):
                probabilities = probabilities.numpy()
            probabilities = np.array(probabilities).flatten().tolist()
            
            # Create simple bar chart with better value representation
            percentages = [p * 100 for p in probabilities]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=percentages,
                    y=class_names,
                    orientation='h',
                    text=[f'{p:.3f}%' if p < 1 else f'{p:.1f}%' for p in percentages],
                    textposition='outside',
                    marker=dict(color=[class_colors.get(cls, '#1f77b4') for cls in class_names])
                )
            ])
            
            fig.update_layout(
                title="Prediction Confidence (Fallback)",
                xaxis_title="Confidence (%)",
                yaxis_title="Cell Type",
                height=300,
                margin=dict(l=80, r=50, t=40, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating fallback chart: {str(e)}")
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
            probabilities = prediction_result['probabilities']
            
            # Create comprehensive data
            df_data = []
            for i, cls in enumerate(Config.CLASS_NAMES):
                prob = probabilities[i]
                percentage = prob * 100
                
                # Format percentage with appropriate precision
                if percentage < 0.01:
                    pct_str = f"{percentage:.4f}%"
                elif percentage < 1:
                    pct_str = f"{percentage:.3f}%"
                else:
                    pct_str = f"{percentage:.2f}%"
                
                df_data.append({
                    'Cell Type': cls,
                    'Raw Probability': f"{prob:.6f}",
                    'Percentage': pct_str,
                    'Confidence Level': VisualizationManager._get_confidence_level(prob),
                    'Description': Config.CLASS_DESCRIPTIONS[cls]
                })
            
            df = pd.DataFrame(df_data)
            
            # Add sorting helper and sort by probability
            df['Sort_Value'] = probabilities
            df = df.sort_values('Sort_Value', ascending=False)
            df = df.drop('Sort_Value', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating results DataFrame: {str(e)}")
            return None
    
    @staticmethod
    def _get_confidence_level(probability):
        """Get confidence level description"""
        if not CONFIG_AVAILABLE:
            return "Unknown"
            
        if probability >= Config.HIGH_CONFIDENCE_THRESHOLD:
            return "Very High"
        elif probability >= Config.GOOD_CONFIDENCE_THRESHOLD:
            return "Good"
        elif probability >= Config.MODERATE_CONFIDENCE_THRESHOLD:
            return "Moderate"
        else:
            return "Low"