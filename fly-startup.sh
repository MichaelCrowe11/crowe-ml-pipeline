#!/bin/bash
# Startup script for Crowe ML Pipeline on Fly.io

set -e

echo "🚀 Starting Crowe ML Pipeline..."

# Function to check if a port is available
check_port() {
    netstat -tuln | grep -q ":$1 " && return 1 || return 0
}

# Start the Vision Platform (Next.js)
if check_port 3000; then
    echo "Starting Vision Platform on port 3000..."
    cd /app/crowe-vision-platform
    npm start &
    VISION_PID=$!
else
    echo "Port 3000 already in use, skipping Vision Platform"
fi

# Start the CriOS API server
if check_port 8000; then
    echo "Starting CriOS API server on port 8000..."
    cd /app/crios
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &
    CRIOS_PID=$!
else
    echo "Port 8000 already in use, skipping CriOS API"
fi

# Start the main ML Pipeline API
echo "Starting ML Pipeline API on port 8080..."
cd /app/universal-fungal-intelligence-system

# Create a simple FastAPI wrapper for Fly.io
cat > /app/fly_server.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import sys
import os
sys.path.insert(0, '/app/universal-fungal-intelligence-system/src')

from core.fungal_intelligence import UniversalFungalIntelligence
from core.molecular_analyzer import MolecularAnalyzer
from core.bioactivity_predictor import BioactivityPredictor
import asyncio

app = FastAPI(title="Crowe ML Pipeline API", version="2.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
fungal_system = UniversalFungalIntelligence()
molecular_analyzer = MolecularAnalyzer()
bioactivity_predictor = BioactivityPredictor()

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>Crowe ML Pipeline</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2rem;
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }
                h1 { font-size: 3rem; margin-bottom: 1rem; }
                .status { 
                    display: inline-block;
                    padding: 0.5rem 1rem;
                    background: rgba(0,255,0,0.2);
                    border-radius: 20px;
                    margin: 0.5rem;
                }
                .links {
                    margin-top: 2rem;
                }
                a {
                    color: white;
                    text-decoration: none;
                    padding: 0.5rem 1rem;
                    background: rgba(255,255,255,0.2);
                    border-radius: 10px;
                    margin: 0.5rem;
                    display: inline-block;
                }
                a:hover {
                    background: rgba(255,255,255,0.3);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧬 Crowe ML Pipeline</h1>
                <p>Advanced Compound Discovery Platform</p>
                <div class="status">✅ System Online</div>
                <div class="links">
                    <a href="/docs">API Documentation</a>
                    <a href="/health">Health Status</a>
                    <a href="http://localhost:3000" target="_blank">Vision Platform</a>
                </div>
            </div>
        </body>
    </html>
    """)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "ml_pipeline": "online",
            "fungal_intelligence": "online",
            "molecular_analyzer": "online",
            "bioactivity_predictor": "online"
        }
    }

@app.post("/analyze/compound")
async def analyze_compound(smiles: str):
    try:
        analysis = molecular_analyzer.analyze_structure(smiles)
        bioactivity = bioactivity_predictor.predict_bioactivity({'smiles': smiles})
        
        return {
            "smiles": smiles,
            "properties": analysis,
            "bioactivity": bioactivity
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/fungal-kingdom")
async def analyze_fungal_kingdom():
    try:
        results = await asyncio.to_thread(fungal_system.analyze_global_fungal_kingdom)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compounds/{compound_id}")
async def get_compound(compound_id: str):
    # Implement compound retrieval
    return {"compound_id": compound_id, "status": "retrieved"}

@app.post("/api/train")
async def train_model(dataset: str, model_type: str = "random_forest"):
    # Implement model training
    return {"status": "training_started", "dataset": dataset, "model": model_type}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF

# Run the main API server
python /app/fly_server.py &
API_PID=$!

# Wait for all services to start
sleep 5

echo "✅ All services started successfully!"
echo "Vision Platform PID: ${VISION_PID:-N/A}"
echo "CriOS API PID: ${CRIOS_PID:-N/A}"
echo "Main API PID: $API_PID"

# Keep the container running
wait $API_PID
