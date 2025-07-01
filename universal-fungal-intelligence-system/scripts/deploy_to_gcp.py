#!/usr/bin/env python3
"""Deploy and run the Universal Fungal Intelligence System on GCP."""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

# GCP Project Configuration
PROJECT_ID = "crowechem-fungi"
BUCKET_NAME = "crowechem-fungi-ml-metrics"
REGION = "us-central1"

def setup_gcp_auth():
    """Set up GCP authentication."""
    print("🔐 Setting up GCP authentication...")
    
    # Check if already authenticated
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        active_accounts = json.loads(result.stdout)
        if active_accounts:
            print(f"✓ Already authenticated as: {active_accounts[0]['account']}")
            return True
    
    # Authenticate
    print("Please authenticate with your Google Cloud account...")
    subprocess.run(["gcloud", "auth", "login"], check=True)
    
    # Set project
    subprocess.run(["gcloud", "config", "set", "project", PROJECT_ID], check=True)
    print(f"✓ Set project to: {PROJECT_ID}")
    
    return True

def enable_apis():
    """Enable required GCP APIs."""
    print("\n🚀 Enabling required APIs...")
    
    apis = [
        "bigquery.googleapis.com",
        "storage.googleapis.com",
        "cloudbuild.googleapis.com"
    ]
    
    for api in apis:
        print(f"  Enabling {api}...")
        subprocess.run(
            ["gcloud", "services", "enable", api, "--project", PROJECT_ID],
            check=True
        )
    
    print("✓ All APIs enabled")

def create_service_account():
    """Create a service account for the application."""
    print("\n👤 Setting up service account...")
    
    service_account_name = "fungal-intelligence-sa"
    service_account_email = f"{service_account_name}@{PROJECT_ID}.iam.gserviceaccount.com"
    
    # Check if service account exists
    result = subprocess.run(
        ["gcloud", "iam", "service-accounts", "describe", service_account_email,
         "--project", PROJECT_ID],
        capture_output=True
    )
    
    if result.returncode != 0:
        # Create service account
        subprocess.run([
            "gcloud", "iam", "service-accounts", "create", service_account_name,
            "--display-name", "Fungal Intelligence System Service Account",
            "--project", PROJECT_ID
        ], check=True)
        
        # Grant permissions
        roles = [
            "roles/bigquery.dataEditor",
            "roles/bigquery.jobUser",
            "roles/storage.objectAdmin"
        ]
        
        for role in roles:
            subprocess.run([
                "gcloud", "projects", "add-iam-policy-binding", PROJECT_ID,
                "--member", f"serviceAccount:{service_account_email}",
                "--role", role
            ], check=True)
        
        print(f"✓ Created service account: {service_account_email}")
    else:
        print(f"✓ Service account already exists: {service_account_email}")
    
    # Create and download key
    key_path = "gcp-service-key.json"
    if not os.path.exists(key_path):
        subprocess.run([
            "gcloud", "iam", "service-accounts", "keys", "create", key_path,
            "--iam-account", service_account_email,
            "--project", PROJECT_ID
        ], check=True)
        print(f"✓ Created service account key: {key_path}")
    
    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_path)
    print(f"✓ Set GOOGLE_APPLICATION_CREDENTIALS")
    
    return service_account_email

def setup_bigquery():
    """Set up BigQuery dataset and tables."""
    print("\n📊 Setting up BigQuery...")
    
    # Create dataset
    subprocess.run([
        "bq", "mk", "--dataset",
        "--project_id", PROJECT_ID,
        "--location", REGION,
        "crowe_ml_pipeline"
    ], capture_output=True)
    
    print("✓ BigQuery dataset ready")

def setup_gcs():
    """Set up Google Cloud Storage bucket."""
    print("\n🗄️  Setting up Cloud Storage...")
    
    # Create bucket
    subprocess.run([
        "gsutil", "mb", "-p", PROJECT_ID, "-l", REGION,
        f"gs://{BUCKET_NAME}"
    ], capture_output=True)
    
    print(f"✓ Storage bucket ready: gs://{BUCKET_NAME}")

def run_analysis(export_to_bigquery: bool = True):
    """Run the fungal intelligence analysis."""
    print("\n🧬 Starting fungal intelligence analysis...")
    
    # Change to the universal-fungal-intelligence-system directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    src_dir = os.path.join(project_dir, "src")
    
    # Add src to Python path
    sys.path.insert(0, src_dir)
    
    # Import and run
    from main import main
    
    try:
        results = main(export_to_bigquery=export_to_bigquery)
        
        print("\n✅ Analysis complete!")
        print(f"   Species analyzed: {results.get('total_species_analyzed', 0)}")
        print(f"   Compounds analyzed: {results.get('total_compounds_analyzed', 0)}")
        print(f"   Breakthrough discoveries: {len(results.get('breakthrough_discoveries', []))}")
        
        if export_to_bigquery:
            print(f"\n📊 View results in BigQuery:")
            print(f"   https://console.cloud.google.com/bigquery?project={PROJECT_ID}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Deploy Fungal Intelligence System to GCP')
    parser.add_argument('--skip-auth', action='store_true', help='Skip authentication setup')
    parser.add_argument('--skip-setup', action='store_true', help='Skip GCP resource setup')
    parser.add_argument('--no-export', action='store_true', help='Run analysis without BigQuery export')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍄 Universal Fungal Intelligence System - GCP Deployment")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        if not args.skip_auth:
            setup_gcp_auth()
            enable_apis()
            create_service_account()
        
        if not args.skip_setup:
            setup_bigquery()
            setup_gcs()
        
        # Run the analysis
        run_analysis(export_to_bigquery=not args.no_export)
        
        print("\n🎉 Deployment and analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 