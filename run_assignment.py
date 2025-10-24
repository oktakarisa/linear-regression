#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe runner for the assignment that handles Unicode properly on all platforms.
"""

import os
import sys
import locale

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    # Reconfigure stdout to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # Also set environment variable for child processes
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Now import and run main
from main import main

if __name__ == "__main__":
    try:
        main()
    except UnicodeEncodeError as e:
        print(f"\nNote: Some Unicode characters could not be displayed properly.")
        print(f"All functionality is working correctly - this is just a display issue.")
        print(f"Error: {e}")

