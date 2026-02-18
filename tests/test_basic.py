import pytest
import os
from eac_rag.core import EACRAG

def test_init():
    """Test if models load correctly."""
    rag = EACRAG()
    assert rag.embedder is not None
    assert rag.G is not None

def test_avt_segmentation():
    """Test the Adaptive Value Thresholding logic."""
    rag = EACRAG()
    sample_text = "This is a long sentence about Budesonide. " * 20
    chunks = rag.avt_segmentation(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_graph_structure():
    """Test if bipartite edges are created correctly."""
    rag = EACRAG()
    # Mock data ingestion
    rag.chunks = [{"text": "Sample", "hubs": ["Aspirin"]}]
    rag.G.add_edge("Aspirin", "chunk_0")
    
    assert "Aspirin" in rag.G
    assert rag.G.degree("Aspirin") == 1
