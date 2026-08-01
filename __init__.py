"""ComfyUI custom nodes for direct, in-process MiniMax-H3 inference.

This package deliberately does not import or communicate with SGLang.  The
large runtime dependencies are imported lazily by the individual loader nodes.
"""

from .minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
