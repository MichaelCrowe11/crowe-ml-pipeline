from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from src.analysis import chemical_analysis, molecular_modifications, therapeutic_assessment, impact_evaluation

router = APIRouter()

@router.post("/analyze/chemical", response_model=Dict[str, Any])
async def analyze_chemical_composition(species_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        analysis_results = await chemical_analysis.analyze_species_chemistry(species_data)
        return analysis_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/modifications", response_model=Dict[str, Any])
async def predict_molecular_modifications(compound_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        modifications_results = await molecular_modifications.predict_molecular_modifications(compound_data)
        return modifications_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/therapeutic", response_model=Dict[str, Any])
async def assess_therapeutic_potential(compound_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        therapeutic_results = await therapeutic_assessment.assess_therapeutic_potential(compound_data)
        return therapeutic_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/impact", response_model=Dict[str, Any])
async def evaluate_impact(discovery_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        impact_results = await impact_evaluation.evaluate_impact(discovery_data)
        return impact_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))