"""
Deployment script for Blood Cell Classification App
Run this from the main project directory
"""

import subprocess
import sys
import os

def check_model_file():
    """Check if model file exists"""
    model_path = "../models/mobilenetv2_cbam_best.h5"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the trained model is in the models/ directory")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    try:
        # Change to parent directory to run app
        os.chdir("..")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main deployment function"""
    print("🩸 Blood Cell Classification App - Deployment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    # Check model file
    if not check_model_file():
        return
    
    # Install requirements
    install_choice = input("\nInstall/update requirements? (y/n): ").lower().strip()
    if install_choice in ['y', 'yes']:
        if not install_requirements():
            return
    
    # Run app
    print("\n🚀 Ready to launch!")
    input("Press Enter to start the Streamlit app...")
    run_streamlit()

if __name__ == "__main__":
    main()