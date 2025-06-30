from typing import Dict, Any

class CompoundProcessor:
    """
    Class for processing compound data collected from various sources.
    """

    @staticmethod
    def normalize_compound_data(compound_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize compound data to ensure consistency across different sources.

        Args:
            compound_data: Raw compound data from various sources.

        Returns:
            Normalized compound data.
        """
        normalized_data = {
            'name': compound_data.get('name', '').strip(),
            'formula': compound_data.get('formula', '').strip(),
            'molecular_weight': compound_data.get('molecular_weight', 0.0),
            'bioactivity': compound_data.get('bioactivity', {}),
            'source': compound_data.get('source', '').strip()
        }
        return normalized_data

    @staticmethod
    def filter_compounds_by_activity(compound_list: List[Dict[str, Any]], min_activity: float) -> List[Dict[str, Any]]:
        """
        Filter compounds based on a minimum bioactivity threshold.

        Args:
            compound_list: List of compound data dictionaries.
            min_activity: Minimum bioactivity score for filtering.

        Returns:
            List of compounds that meet the bioactivity criteria.
        """
        return [compound for compound in compound_list if compound.get('bioactivity', {}).get('score', 0) >= min_activity]

    @staticmethod
    def categorize_compounds(compound_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize compounds based on their chemical classes.

        Args:
            compound_list: List of compound data dictionaries.

        Returns:
            Dictionary categorizing compounds by their chemical class.
        """
        categorized = {}
        for compound in compound_list:
            chemical_class = compound.get('chemical_class', 'Unknown')
            if chemical_class not in categorized:
                categorized[chemical_class] = []
            categorized[chemical_class].append(compound)
        return categorized

    @staticmethod
    def summarize_compound_data(compound_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of compound data including total count and average molecular weight.

        Args:
            compound_list: List of compound data dictionaries.

        Returns:
            Summary of compound data.
        """
        total_count = len(compound_list)
        average_weight = sum(compound.get('molecular_weight', 0) for compound in compound_list) / total_count if total_count > 0 else 0
        return {
            'total_count': total_count,
            'average_molecular_weight': average_weight
        }