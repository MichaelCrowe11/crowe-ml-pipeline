import os
import subprocess
import sys

def deploy_application():
    """Deploy the Universal Fungal Intelligence System application."""
    
    # Step 1: Install dependencies
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Step 2: Run database migrations
    print("Running database migrations...")
    subprocess.check_call([sys.executable, "src/database/migrations/run_migrations.py"])
    
    # Step 3: Start the application
    print("Starting the application...")
    subprocess.Popen([sys.executable, "src/main.py"])

    print("Deployment completed successfully.")

if __name__ == "__main__":
    deploy_application()