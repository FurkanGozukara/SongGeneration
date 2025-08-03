#!/usr/bin/env python
"""
Quiet version of generate.py that suppresses all verbose output.
Usage: python generate_quiet.py [same arguments as generate.py]
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and configure quiet mode before importing generate
from utils.suppress_output import disable_verbose_logging, suppress_output

# Disable verbose logging
disable_verbose_logging()

# Import and run generate with output suppression
with suppress_output():
    import generate
    
# The generate module will handle its own argument parsing and execution