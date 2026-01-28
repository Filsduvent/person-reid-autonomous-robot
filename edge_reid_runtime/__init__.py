from edge_reid_runtime.core.interfaces import (
    Detection,
    Embedder,
    Frame,
    GalleryManager,
    IdentityAssigner,
    InputSource,
    Profiler,
    Tracker,
    Visualizer,
)
from edge_reid_runtime.core.types import RunConfig, validate_run_config
from edge_reid_runtime.sources.factory import create_source

__all__ = [
    "__version__",
    "create_source",
    "Detection",
    "Embedder",
    "Frame",
    "GalleryManager",
    "IdentityAssigner",
    "InputSource",
    "Profiler",
    "RunConfig",
    "Tracker",
    "Visualizer",
    "validate_run_config",
]

__version__ = "0.1.0"
