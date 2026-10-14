"""Alias for the V4 scope-aware synthetic scaffold checks."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("test_scopes_v4.py")),run_name="__main__")
