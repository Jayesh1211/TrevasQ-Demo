import streamlit as st
import subprocess
import os
import sys
import importlib.util
import time

# Configure the page
st.set_page_config(
    page_title="Demo Selector",
    page_icon="🚀",
    layout="wide"
)

# Define the demos information
demos = [
    {
        "name": "Demo 1",
        "file_name": "integrated-quantum-evaluator(1).py",
        "description": "Description of demo 1",
        "requirements": "requirements-QRA.txt"
    },
    {
        "name": "Demo 2",
        "file_name": "WDEMO_(1).py",
        "description": "Description of demo 2",
        "requirements": "requirements-QImage.txt"
    },
    {
        "name": "Demo 3",
        "file_name": "xyz.py",
        "description": "Description of demo 3",
        "requirements": "requirements-Genomics.txt"
    }
]

def run_demo(demo_file):
    """Run a selected demo file directly in the current Streamlit app"""
    try:
        with st.spinner(f"Loading {demo_file}..."):
            # Get the absolute path to the demo file
            demo_path = os.path.abspath(demo_file)
            
            # Import the module
            spec = importlib.util.spec_from_file_location("demo_module", demo_path)
            if spec is None:
                st.error(f"Could not load module from {demo_path}")
                return False
            
            # Load and execute the module
            demo_module = importlib.util.module_from_spec(spec)
            sys.modules["demo_module"] = demo_module
            spec.loader.exec_module(demo_module)
            
            # If the module defines a main function, call it
            if hasattr(demo_module, "main"):
                demo_module.main()
                
            return True
    except Exception as e:
        st.error(f"Error loading demo: {e}")
        return False

def install_requirements(requirements_file):
    """Install required packages for a specific demo"""
    try:
        if os.path.exists(requirements_file):
            with st.spinner(f"Installing requirements from {requirements_file}..."):
                # Try to install with --user flag first
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--user", "-r", requirements_file
                    ])
                    st.success(f"Requirements installed successfully!")
                    return True
                except subprocess.CalledProcessError:
                    # If that fails, show instructions for using a virtual environment
                    st.error("""
                    Permission error when installing packages. Please run this app in a virtual environment:
                    
                    ```
                    # Create a virtual environment
                    python -m venv myenv
                    
                    # Activate it
                    source myenv/bin/activate  # On Linux/Mac
                    # or
                    myenv\\Scripts\\activate  # On Windows
                    
                    # Install Streamlit
                    pip install streamlit
                    
                    # Run this app
                    streamlit run demo_selector.py
                    ```
                    """)
                    return False
        else:
            st.warning(f"Requirements file {requirements_file} not found.")
            return True  # Continue anyway
    except Exception as e:
        st.error(f"Failed to install requirements: {e}")
        return False

def main():
    # Store navigation state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Home page - Demo selection
    if st.session_state.page == "home":
        st.title("🚀 My Streamlit Demos")
        st.markdown("Select a demo to run from the options below:")
        
        # Create columns for demo cards
        cols = st.columns(len(demos))
        
        for i, demo in enumerate(demos):
            with cols[i]:
                st.subheader(demo["name"])
                st.write(demo["description"])
                
                # Show option to install requirements separately
                if st.button(f"Install Requirements", key=f"install_{i}"):
                    install_requirements(demo["requirements"])
                
                if st.button(f"Run {demo['name']}", key=f"run_{i}"):
                    # Store the selected demo index
                    st.session_state.selected_demo = i
                    st.session_state.page = "demo"
                    st.rerun()
    
    # Demo page - Running the selected demo
    elif st.session_state.page == "demo" and "selected_demo" in st.session_state:
        demo = demos[st.session_state.selected_demo]
        
        # Add a back button
        if st.button("← Back to Demo Selection"):
            st.session_state.page = "home"
            # Clear any stored state for the demo
            if "selected_demo" in st.session_state:
                del st.session_state.selected_demo
            st.rerun()
        
        st.title(f"🚀 {demo['name']}")
        
        # Option to bypass requirements installation
        skip_requirements = st.checkbox("Skip requirements installation (if already installed)")
        
        if skip_requirements or install_requirements(demo["requirements"]):
            # Run the demo
            run_demo(demo["file_name"])

if __name__ == "__main__":
    main()
