import streamlit as st
import subprocess
import os
import sys
import signal
import psutil
import glob
from pathlib import Path
import importlib.util

# Configure the page
st.set_page_config(
    page_title="Demo Hub",
    page_icon="🚀",
    layout="wide"
)

# Define the demo information (replace with your actual GitHub repos and demos)
demos = [
    {
        "name": "Demo 1",
        "description": "First project demo",
        "repo_url": "https://github.com/yourusername/project1",
        "script_path": "path/to/your/demo1.py"  # Relative to this file
    },
    {
        "name": "Demo 2",
        "description": "Second project demo",
        "repo_url": "https://github.com/yourusername/project2",
        "script_path": "path/to/your/demo2.py"
    },
    {
        "name": "Demo 3",
        "description": "Third project demo",
        "repo_url": "https://github.com/yourusername/project3",
        "script_path": "path/to/your/demo3.py"
    }
]

def load_demo_directly(script_path):
    """Load and run a Streamlit script directly within the current Streamlit app."""
    try:
        # Get the absolute path
        abs_path = os.path.abspath(script_path)
        
        # Import the module
        spec = importlib.util.spec_from_file_location("demo_module", abs_path)
        if spec is None:
            st.error(f"Could not load module from {abs_path}")
            return False
        
        demo_module = importlib.util.module_from_spec(spec)
        sys.modules["demo_module"] = demo_module
        spec.loader.exec_module(demo_module)
        
        return True
    except Exception as e:
        st.error(f"Error loading demo: {e}")
        return False

def run_demo_as_subprocess(script_path, port):
    """Run a Streamlit script as a subprocess on a specific port."""
    try:
        # Start the demo on a specified port
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        st.session_state.active_process = process.pid
        st.session_state.port = port
        return True
    except Exception as e:
        st.error(f"Error launching demo: {e}")
        return False

def terminate_subprocess():
    """Terminate the running subprocess if it exists."""
    if "active_process" in st.session_state:
        try:
            process = psutil.Process(st.session_state.active_process)
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()
            st.session_state.pop("active_process")
            st.session_state.pop("port", None)
        except (psutil.NoSuchProcess, ProcessLookupError):
            # Process already terminated
            pass

def main():
    # Title and introduction
    st.title("🚀 My Streamlit Demo Hub")
    st.markdown("Welcome to my Streamlit Demo Hub! Select a demo to explore.")
    
    # Display the demos in cards
    cols = st.columns(len(demos))
    
    for i, demo in enumerate(demos):
        with cols[i]:
            st.subheader(demo["name"])
            st.write(demo["description"])
            
            # GitHub repo link
            st.markdown(f"[GitHub Repository]({demo['repo_url']})")
            
            # Launch button
            if st.button(f"Launch {demo['name']}", key=f"launch_{i}"):
                # Clear session state if another demo was running
                if "active_demo" in st.session_state:
                    # Terminate any running subprocess
                    terminate_subprocess()
                    
                # Set the active demo
                st.session_state.active_demo = i
                st.rerun()
    
    # Display the selected demo
    if "active_demo" in st.session_state:
        demo_index = st.session_state.active_demo
        selected_demo = demos[demo_index]
        
        st.markdown("---")
        st.subheader(f"Running: {selected_demo['name']}")
        
        # Check if we should load directly or as a subprocess
        if "active_process" not in st.session_state:
            # Option 1: Run as an iframe to a subprocess (recommended for isolation)
            port = 8501 + demo_index + 1  # Use different ports for each demo
            success = run_demo_as_subprocess(selected_demo["script_path"], port)
            
            if success:
                st.info(f"Demo is running on port {port}")
                st.components.v1.iframe(
                    src=f"http://localhost:{port}",
                    height=600,
                    scrolling=True
                )
            
            # Option 2: Alternative approach - load the demo directly
            # Uncomment the following line to use direct loading instead of subprocess
            # load_demo_directly(selected_demo["script_path"])
        else:
            # Show iframe to the running demo
            st.components.v1.iframe(
                src=f"http://localhost:{st.session_state.port}",
                height=600,
                scrolling=True
            )
        
        # Button to go back to the hub
        if st.button("← Back to Demo Hub"):
            terminate_subprocess()
            st.session_state.pop("active_demo", None)
            st.rerun()

if __name__ == "__main__":
    main()

    # Clean up subprocess when the app is closed
    def cleanup():
        terminate_subprocess()
    
    # Register cleanup function to be called on exit
    import atexit
    atexit.register(cleanup)
