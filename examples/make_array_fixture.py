"""Write a small multirate trace for the Forager NPZ converter."""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
with args.output.open("xb") as stream:
    np.savez(
        stream,
        rewards=np.array([0.0, 1.0, 1.0, 1.0]),
        total_loss=np.array([0.5, 0.25]),
        gradient_norm=np.array([0.8, 0.4]),
        pos=np.array([[0, 0], [0, 1], [1, 1], [1, 2]]),
        biome_id=np.array([0, 0, 1, 1]),
        object_collected_id=np.array([-1, 0, 1, 2]),
    )
