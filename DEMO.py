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
        "description": "Description of demo 1"
        
    },
    {
        "name": "Demo 2",
        "file_name": "WDEMO_(1).py",
        "description": "Description of demo 2"
       
    },
    {
        "name": "Demo 3",
        "file_name": "xyz.py",
        "description": "Description of demo 3"
        
    }
]


# Path to the single requirements file
REQUIREMENTS_FILE = "requirements.txt"

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
        st.error(f"Traceback: {str(e)}")
        return False

def main():
    # Store navigation state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Home page - Demo selection
    if st.session_state.page == "home":
        st.title("🚀 My Streamlit Demos")
        st.markdown("Select a demo to run from the options below:")
        
        st.info("""
        **Note:** Before running the demos, make sure all required packages are installed.
        
        If you're running into permission errors, we recommend using a virtual environment:
        ```
        # Create a virtual environment
        python -m venv myenv
        
        # Activate it
        source myenv/bin/activate  # On Linux/Mac
        # OR
        myenv\\Scripts\\activate    # On Windows
        
        # Install all requirements
        pip install -r requirements.txt
        
        # Run this app
        streamlit run demo_selector.py
        ```
        """)
        
        # Create columns for demo cards
        cols = st.columns(len(demos))
        
        for i, demo in enumerate(demos):
            with cols[i]:
                st.subheader(demo["name"])
                st.write(demo["description"])
                
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
        
        # Run the demo
        run_demo(demo["file_name"])

if __name__ == "__main__":
    main()

