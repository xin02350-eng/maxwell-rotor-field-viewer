"""Alias for the V5 scope-aware synthetic scaffold checks."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("test_scopes_v5.py")),run_name="__main__")
