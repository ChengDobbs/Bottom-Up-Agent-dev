#!/usr/bin/env python3
"""
Standalone script to run the enhanced visualizer with skills display and interactive control
Usage: python run_visualizer.py --config_file "config/sts_omni_claude.yaml" --port 8051
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the visualizer
if __name__ == '__main__':
    from BottomUpAgent.visualizer import main
    main()