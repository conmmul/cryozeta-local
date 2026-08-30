"""CryoZeta local web server.

A local-only FastAPI wrapper around an existing CryoZeta installation. It
discovers the CryoZeta checkout, Pixi environment and GPUs at runtime, so the
same checkout runs unmodified on any workstation.
"""

__version__ = "0.1.0"
