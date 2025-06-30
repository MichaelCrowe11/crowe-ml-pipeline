import asyncio
from src.core.fungal_intelligence import UniversalFungalIntelligence

async def main():
    fungal_intelligence_system = UniversalFungalIntelligence()
    analysis_results = await fungal_intelligence_system.analyze_global_fungal_kingdom()
    
    # Here you can handle the results, e.g., save to a file or print
    print(analysis_results)

if __name__ == "__main__":
    asyncio.run(main())