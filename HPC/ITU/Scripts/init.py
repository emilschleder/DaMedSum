# utils.py
import sys
import os

def add_hpc_to_sys_path():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def add_config_to_sys_path():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))