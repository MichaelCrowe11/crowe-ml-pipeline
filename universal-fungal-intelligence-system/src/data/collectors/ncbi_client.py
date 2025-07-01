import requests
import xml.etree.ElementTree as ET
import logging
import time
from typing import Dict, Any, List, Optional
import json
from urllib.parse import quote

logger = logging.getLogger(__name__)

class NCBIClient:
    """
    Client for interacting with NCBI databases via E-utilities API.
    Provides access to fungal genomes, proteins, and literature.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NCBI client.
        
        Args:
            api_key: NCBI API key for increased rate limits
        """
        self.api_key = api_key
        self.rate_limit_delay = 0.4 if api_key else 0.4  # 3 requests/second without key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniversalFungalIntelligenceSystem/1.0'
        })
    
    def search_fungi(self, query: str, database: str = "taxonomy", max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search NCBI databases for fungal data.
        
        Args:
            query: Search query
            database: NCBI database (taxonomy, protein, nuccore, pubmed)
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Build search URL
            search_params = {
                'db': database,
                'term': f'"{query}" AND fungi[orgn]',
                'retmax': max_results,
                'retmode': 'json'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            # Search for IDs
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            search_results = response.json()
            id_list = search_results.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                logger.info(f"No results found for query: {query}")
                return []
            
            # Fetch detailed data for each ID
            results = []
            for uid in id_list:
                time.sleep(self.rate_limit_delay)
                details = self.fetch_details(uid, database)
                if details:
                    results.append(details)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching NCBI: {str(e)}")
            return []
    
    def fetch_species_details(self, species_id: str) -> Dict[str, Any]:
        """
        Fetch detailed information about a fungal species from NCBI Taxonomy.
        
        Args:
            species_id: NCBI Taxonomy ID
            
        Returns:
            Dict with species details
        """
        try:
            # Fetch taxonomy data
            fetch_params = {
                'db': 'taxonomy',
                'id': species_id,
                'retmode': 'xml'
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            response = self.session.get(fetch_url, params=fetch_params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            taxon = root.find('.//Taxon')
            
            if taxon is None:
                return {}
            
            species_data = {
                'ncbi_taxid': species_id,
                'scientific_name': taxon.findtext('ScientificName', ''),
                'common_name': taxon.findtext('CommonName', ''),
                'rank': taxon.findtext('Rank', ''),
                'division': taxon.findtext('Division', ''),
                'lineage': self._extract_lineage(taxon),
                'genetic_code': taxon.findtext('GeneticCode/GCId', ''),
                'mitochondrial_genetic_code': taxon.findtext('MitoGeneticCode/MGCId', ''),
                'other_names': self._extract_other_names(taxon),
                'pubmed_ids': self._get_related_publications(species_id)
            }
            
            # Get protein and genome data
            species_data['proteins'] = self._get_protein_count(species_id)
            species_data['genomes'] = self._get_genome_info(species_id)
            
            return species_data
            
        except Exception as e:
            logger.error(f"Error fetching species details for {species_id}: {str(e)}")
            return {}
    
    def fetch_fungal_proteins(self, species_name: str, max_proteins: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch protein sequences for a fungal species.
        
        Args:
            species_name: Name of the fungal species
            max_proteins: Maximum number of proteins to fetch
            
        Returns:
            List of protein data
        """
        try:
            # Search for proteins
            proteins = self.search_fungi(
                f"{species_name} AND biomol_mrna[PROP]",
                database="protein",
                max_results=max_proteins
            )
            
            protein_list = []
            for protein in proteins:
                protein_data = {
                    'accession': protein.get('accession', ''),
                    'title': protein.get('title', ''),
                    'organism': protein.get('organism', ''),
                    'length': protein.get('length', 0),
                    'molecular_weight': protein.get('molecular_weight', 0),
                    'sequence': protein.get('sequence', ''),
                    'function': protein.get('function', ''),
                    'gene_names': protein.get('gene_names', [])
                }
                protein_list.append(protein_data)
            
            return protein_list
            
        except Exception as e:
            logger.error(f"Error fetching proteins for {species_name}: {str(e)}")
            return []
    
    def fetch_fungal_metabolite_literature(self, compound_name: str, species_name: str = "") -> List[Dict[str, Any]]:
        """
        Search PubMed for literature about fungal metabolites.
        
        Args:
            compound_name: Name of the metabolite
            species_name: Optional species name
            
        Returns:
            List of relevant publications
        """
        try:
            # Build PubMed query
            query_parts = [f'"{compound_name}"', 'fungi', 'metabolite']
            if species_name:
                query_parts.append(f'"{species_name}"')
            
            query = ' AND '.join(query_parts)
            
            # Search PubMed
            publications = self.search_fungi(query, database="pubmed", max_results=20)
            
            pub_list = []
            for pub in publications:
                pub_data = {
                    'pmid': pub.get('pmid', ''),
                    'title': pub.get('title', ''),
                    'authors': pub.get('authors', []),
                    'journal': pub.get('journal', ''),
                    'year': pub.get('year', ''),
                    'abstract': pub.get('abstract', ''),
                    'keywords': pub.get('keywords', []),
                    'doi': pub.get('doi', '')
                }
                pub_list.append(pub_data)
            
            return pub_list
            
        except Exception as e:
            logger.error(f"Error fetching literature for {compound_name}: {str(e)}")
            return []
    
    def fetch_details(self, uid: str, database: str) -> Dict[str, Any]:
        """Fetch detailed information for a specific UID."""
        try:
            time.sleep(self.rate_limit_delay)
            
            if database == "taxonomy":
                return self.fetch_species_details(uid)
            elif database == "protein":
                return self._fetch_protein_details(uid)
            elif database == "pubmed":
                return self._fetch_pubmed_details(uid)
            else:
                return {'uid': uid, 'database': database}
                
        except Exception as e:
            logger.error(f"Error fetching details for {uid}: {str(e)}")
            return {}
    
    def _fetch_protein_details(self, protein_id: str) -> Dict[str, Any]:
        """Fetch detailed protein information."""
        try:
            fetch_params = {
                'db': 'protein',
                'id': protein_id,
                'rettype': 'gb',
                'retmode': 'xml'
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            response = self.session.get(fetch_url, params=fetch_params)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.text)
            seq_entry = root.find('.//GBSeq')
            
            if seq_entry is None:
                return {}
            
            return {
                'accession': seq_entry.findtext('GBSeq_primary-accession', ''),
                'title': seq_entry.findtext('GBSeq_definition', ''),
                'organism': seq_entry.findtext('GBSeq_organism', ''),
                'length': int(seq_entry.findtext('GBSeq_length', '0')),
                'sequence': seq_entry.findtext('GBSeq_sequence', ''),
                'features': self._extract_features(seq_entry)
            }
            
        except Exception as e:
            logger.error(f"Error fetching protein details: {str(e)}")
            return {}
    
    def _fetch_pubmed_details(self, pmid: str) -> Dict[str, Any]:
        """Fetch PubMed article details."""
        try:
            fetch_params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml'
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            response = self.session.get(fetch_url, params=fetch_params)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.text)
            article = root.find('.//MedlineCitation')
            
            if article is None:
                return {}
            
            # Extract article data
            pub_data = {
                'pmid': pmid,
                'title': article.findtext('.//ArticleTitle', ''),
                'abstract': article.findtext('.//AbstractText', ''),
                'journal': article.findtext('.//Journal/Title', ''),
                'year': article.findtext('.//PubDate/Year', ''),
                'authors': self._extract_authors(article),
                'keywords': self._extract_keywords(article)
            }
            
            return pub_data
            
        except Exception as e:
            logger.error(f"Error fetching PubMed details: {str(e)}")
            return {}
    
    def _extract_lineage(self, taxon_element) -> List[str]:
        """Extract taxonomic lineage."""
        lineage = []
        lineage_ex = taxon_element.find('LineageEx')
        if lineage_ex is not None:
            for taxon in lineage_ex.findall('Taxon'):
                name = taxon.findtext('ScientificName')
                if name:
                    lineage.append(name)
        return lineage
    
    def _extract_other_names(self, taxon_element) -> List[str]:
        """Extract alternative names."""
        names = []
        other_names = taxon_element.find('OtherNames')
        if other_names is not None:
            for name_elem in other_names:
                name = name_elem.text
                if name:
                    names.append(name)
        return names
    
    def _extract_features(self, seq_element) -> List[Dict[str, str]]:
        """Extract sequence features."""
        features = []
        feature_table = seq_element.find('GBSeq_feature-table')
        if feature_table is not None:
            for feature in feature_table.findall('GBFeature'):
                feat_data = {
                    'key': feature.findtext('GBFeature_key', ''),
                    'location': feature.findtext('GBFeature_location', '')
                }
                features.append(feat_data)
        return features
    
    def _extract_authors(self, article_element) -> List[str]:
        """Extract author names from article."""
        authors = []
        author_list = article_element.find('.//AuthorList')
        if author_list is not None:
            for author in author_list.findall('Author'):
                last_name = author.findtext('LastName', '')
                fore_name = author.findtext('ForeName', '')
                if last_name:
                    authors.append(f"{last_name}, {fore_name}")
        return authors
    
    def _extract_keywords(self, article_element) -> List[str]:
        """Extract keywords from article."""
        keywords = []
        keyword_list = article_element.find('.//KeywordList')
        if keyword_list is not None:
            for keyword in keyword_list.findall('.//Keyword'):
                if keyword.text:
                    keywords.append(keyword.text)
        return keywords
    
    def _get_related_publications(self, taxid: str) -> List[str]:
        """Get PubMed IDs for publications about this species."""
        try:
            search_params = {
                'db': 'pubmed',
                'term': f'txid{taxid}[Organism]',
                'retmax': 10,
                'retmode': 'json'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            results = response.json()
            return results.get('esearchresult', {}).get('idlist', [])
            
        except Exception:
            return []
    
    def _get_protein_count(self, taxid: str) -> int:
        """Get number of proteins for species."""
        try:
            search_params = {
                'db': 'protein',
                'term': f'txid{taxid}[Organism]',
                'rettype': 'count',
                'retmode': 'json'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            results = response.json()
            return int(results.get('esearchresult', {}).get('count', 0))
            
        except Exception:
            return 0
    
    def _get_genome_info(self, taxid: str) -> Dict[str, Any]:
        """Get genome information for species."""
        try:
            search_params = {
                'db': 'genome',
                'term': f'txid{taxid}[Organism]',
                'retmax': 1,
                'retmode': 'json'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            results = response.json()
            if results.get('esearchresult', {}).get('count', 0) > 0:
                return {
                    'available': True,
                    'genome_ids': results.get('esearchresult', {}).get('idlist', [])
                }
            
            return {'available': False, 'genome_ids': []}
            
        except Exception:
            return {'available': False, 'genome_ids': []}
    
    def close(self):
        """Close the session."""
        self.session.close()