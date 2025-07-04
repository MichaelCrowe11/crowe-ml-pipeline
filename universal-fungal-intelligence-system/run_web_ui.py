#!/usr/bin/env python3
"""
Launch the Universal Fungal Intelligence System Web UI
"""

import os
import sys
import subprocess
import argparse

def main():
    """Launch Streamlit app."""
    parser = argparse.ArgumentParser(description='Launch the Universal Fungal Intelligence System Web UI')
    parser.add_argument('--host', default='localhost', help='Host address (default: localhost)')
    parser.add_argument('--port', type=int, default=8501, help='Port number (default: 8501)')
    parser.add_argument('--docker', action='store_true', help='Running in Docker container')
    
    args = parser.parse_args()
    
    print("🍄 Starting Universal Fungal Intelligence System Web UI...")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docker mode: {args.docker}")
    
    # Get the app path
    app_path = os.path.join(os.path.dirname(__file__), "src", "web_ui", "app.py")
    
    # Check if app file exists
    if not os.path.exists(app_path):
        print(f"❌ App file not found: {app_path}")
        sys.exit(1)
    
    # Configure host for Docker
    if args.docker or os.environ.get('DOCKER_ENV'):
        host = '0.0.0.0'  # Allow external connections in Docker
    else:
        host = args.host
    
    # Run Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            app_path,
            "--server.port", str(args.port),
            "--server.address", host,
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ]
        
        if args.docker:
            cmd.extend(["--server.enableCORS", "false"])
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✋ Web UI stopped by user")
    except Exception as e:
        print(f"❌ Error running web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 