"""Put fpga/host on sys.path so ``import cpl`` works from any rootdir."""
import os
import sys

HOST = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HOST not in sys.path:
    sys.path.insert(0, HOST)
