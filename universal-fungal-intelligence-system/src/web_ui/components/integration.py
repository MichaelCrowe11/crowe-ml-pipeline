"""
Integration Framework for the Universal Fungal Intelligence System

This module provides a flexible framework for integrating additional functionalities
from other applications into the main system.
"""

import streamlit as st
import importlib
import sys
import os
from typing import Dict, Any, List, Callable, Optional
from abc import ABC, abstractmethod

class IntegrationModule(ABC):
    """Abstract base class for integration modules."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the integration module."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of the integration module."""
        pass
    
    @abstractmethod
    def get_icon(self) -> str:
        """Return an emoji icon for the module."""
        pass
    
    @abstractmethod
    def render_interface(self) -> None:
        """Render the module's user interface."""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Return list of required Python packages."""
        pass
    
    def initialize(self) -> bool:
        """Initialize the module. Return True if successful."""
        try:
            # Check dependencies
            missing_deps = self.check_dependencies()
            if missing_deps:
                st.error(f"Missing dependencies: {', '.join(missing_deps)}")
                return False
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize {self.get_name()}: {str(e)}")
            return False
    
    def check_dependencies(self) -> List[str]:
        """Check if all required dependencies are installed."""
        missing = []
        for dep in self.get_dependencies():
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing

class IntegrationManager:
    """Manages integration modules for the Universal Fungal Intelligence System."""
    
    def __init__(self):
        self.modules: Dict[str, IntegrationModule] = {}
        self.active_modules: List[str] = []
    
    def register_module(self, module: IntegrationModule) -> bool:
        """Register a new integration module."""
        try:
            if module.initialize():
                self.modules[module.get_name()] = module
                return True
            return False
        except Exception as e:
            st.error(f"Failed to register module {module.get_name()}: {str(e)}")
            return False
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module names."""
        return list(self.modules.keys())
    
    def get_module(self, name: str) -> Optional[IntegrationModule]:
        """Get a specific module by name."""
        return self.modules.get(name)
    
    def render_module_selector(self) -> Optional[str]:
        """Render module selection interface."""
        if not self.modules:
            st.warning("No integration modules available.")
            return None
        
        module_options = [f"{module.get_icon()} {module.get_name()}" 
                         for module in self.modules.values()]
        
        selected = st.selectbox(
            "Select Integration Module",
            options=module_options,
            help="Choose an additional functionality to integrate"
        )
        
        if selected:
            # Extract module name from the formatted option
            module_name = selected.split(' ', 1)[1]
            return module_name
        
        return None
    
    def render_module_interface(self, module_name: str) -> None:
        """Render the interface for a specific module."""
        module = self.modules.get(module_name)
        if module:
            try:
                module.render_interface()
            except Exception as e:
                st.error(f"Error rendering {module_name}: {str(e)}")
        else:
            st.error(f"Module '{module_name}' not found.")

# Example integration modules for common app types

class DataVisualizationModule(IntegrationModule):
    """Example integration module for data visualization apps."""
    
    def get_name(self) -> str:
        return "Data Visualization"
    
    def get_description(self) -> str:
        return "Advanced data visualization and charting capabilities"
    
    def get_icon(self) -> str:
        return "📊"
    
    def get_dependencies(self) -> List[str]:
        return ["plotly", "matplotlib", "seaborn"]
    
    def render_interface(self) -> None:
        st.markdown("## 📊 Data Visualization Module")
        st.info("This is a placeholder for data visualization integration.")
        
        # Example visualization options
        viz_type = st.selectbox(
            "Visualization Type",
            ["Scatter Plot", "Bar Chart", "Heatmap", "3D Plot"]
        )
        
        if viz_type == "Scatter Plot":
            st.markdown("### Scatter Plot Configuration")
            # Add scatter plot configuration options
        elif viz_type == "Bar Chart":
            st.markdown("### Bar Chart Configuration")
            # Add bar chart configuration options

class MLModelModule(IntegrationModule):
    """Example integration module for machine learning models."""
    
    def get_name(self) -> str:
        return "ML Models"
    
    def get_description(self) -> str:
        return "Additional machine learning models and algorithms"
    
    def get_icon(self) -> str:
        return "🤖"
    
    def get_dependencies(self) -> List[str]:
        return ["scikit-learn", "tensorflow", "pytorch"]
    
    def render_interface(self) -> None:
        st.markdown("## 🤖 Machine Learning Module")
        st.info("This is a placeholder for ML model integration.")
        
        # Example ML options
        model_type = st.selectbox(
            "Model Type",
            ["Classification", "Regression", "Clustering", "Deep Learning"]
        )
        
        if model_type == "Classification":
            st.markdown("### Classification Model Configuration")
            # Add classification model options

class APIIntegrationModule(IntegrationModule):
    """Example integration module for API integrations."""
    
    def get_name(self) -> str:
        return "API Integration"
    
    def get_description(self) -> str:
        return "Integration with external APIs and services"
    
    def get_icon(self) -> str:
        return "🔗"
    
    def get_dependencies(self) -> List[str]:
        return ["requests", "aiohttp"]
    
    def render_interface(self) -> None:
        st.markdown("## 🔗 API Integration Module")
        st.info("This is a placeholder for API integration.")
        
        # Example API options
        api_type = st.selectbox(
            "API Type",
            ["REST API", "GraphQL", "WebSocket", "Database"]
        )
        
        if api_type == "REST API":
            st.markdown("### REST API Configuration")
            # Add REST API configuration options

def create_custom_integration_module(name: str, description: str, icon: str,
                                   dependencies: List[str], 
                                   render_func: Callable[[], None]) -> IntegrationModule:
    """Create a custom integration module dynamically."""
    
    class CustomModule(IntegrationModule):
        def get_name(self) -> str:
            return name
        
        def get_description(self) -> str:
            return description
        
        def get_icon(self) -> str:
            return icon
        
        def get_dependencies(self) -> List[str]:
            return dependencies
        
        def render_interface(self) -> None:
            render_func()
    
    return CustomModule()

def render_integration_help() -> None:
    """Render help information for integration."""
    st.markdown("## 🔧 Integration Help")
    
    with st.expander("How to Integrate Your App"):
        st.markdown("""
        ### Steps to Integrate Your Application:
        
        1. **Create an Integration Module:**
           - Inherit from `IntegrationModule`
           - Implement required methods
           - Define dependencies
        
        2. **Register the Module:**
           - Use `IntegrationManager.register_module()`
           - The module will be automatically available
        
        3. **Test Integration:**
           - Use the module selector
           - Verify all functionalities work
        
        ### Example Integration:
        ```python
        class YourAppModule(IntegrationModule):
            def get_name(self) -> str:
                return "Your App Name"
            
            def get_description(self) -> str:
                return "Description of your app"
            
            def get_icon(self) -> str:
                return "🎯"
            
            def get_dependencies(self) -> List[str]:
                return ["your", "dependencies"]
            
            def render_interface(self) -> None:
                # Your app's UI code here
                st.markdown("## Your App Interface")
        ```
        """)
    
    with st.expander("Common Integration Patterns"):
        st.markdown("""
        ### Data Processing Integration:
        - Import data from your app
        - Apply fungal intelligence analysis
        - Export enhanced results
        
        ### Visualization Integration:
        - Use your app's visualization components
        - Display fungal analysis results
        - Create custom dashboards
        
        ### Model Integration:
        - Combine your ML models with fungal models
        - Create ensemble predictions
        - Cross-validate results
        """)

# Global integration manager instance
integration_manager = IntegrationManager()

# Register default modules
integration_manager.register_module(DataVisualizationModule())
integration_manager.register_module(MLModelModule())
integration_manager.register_module(APIIntegrationModule())
