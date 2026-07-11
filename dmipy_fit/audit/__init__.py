"""Citation graph and methods-section generation for dmipy models."""

from .citations import walk_citation_graph
from .methods_section import generate_methods_section, generate_bibtex

__all__ = [
    'walk_citation_graph',
    'generate_methods_section',
    'generate_bibtex',
]
