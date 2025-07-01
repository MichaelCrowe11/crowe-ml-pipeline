#!/usr/bin/env python3
"""
Launch the Universal Fungal Intelligence System Web UI
"""

import os
import sys
import subprocess

def main():
    """Launch Streamlit app."""
    print("🍄 Starting Universal Fungal Intelligence System Web UI...")
    print("=" * 60)
    
    # Get the app path
    app_path = os.path.join(os.path.dirname(__file__), "src", "web_ui", "app.py")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✋ Web UI stopped by user")
    except Exception as e:
        print(f"❌ Error running web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 