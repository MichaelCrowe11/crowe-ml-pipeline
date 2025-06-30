from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FungalSpecies(Base):
    __tablename__ = 'fungal_species'

    id = Column(Integer, primary_key=True)
    scientific_name = Column(String(255), nullable=False)
    genus = Column(String(100), nullable=False)
    species = Column(String(100), nullable=False)
    phylum = Column(String(100), nullable=False)
    habitat = Column(String(255))
    economic_importance = Column(Text)
    documented_compounds = Column(Text)
    bioactivity_reports = Column(Text)

class ChemicalCompound(Base):
    __tablename__ = 'chemical_compounds'

    id = Column(Integer, primary_key=True)
    compound_name = Column(String(255), nullable=False)
    molecular_formula = Column(String(100))
    molecular_weight = Column(Float)
    log_p = Column(Float)
    tpsa = Column(Float)
    h_donors = Column(Integer)
    h_acceptors = Column(Integer)
    bioactivity = Column(Text)
    synthesis_pathway = Column(Text)