import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_parser():
    parser = argparse.ArgumentParser(description="joblib scaler to npy files")
    parser.add_argument("input_file", type=str, help="input file")
    parser.add_argument("out_dir", type=str, help="out directory")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    input_file = Path(args.input_file)
    scaler = joblib.load(input_file)
    assert isinstance(scaler, StandardScaler)

    print(f"Converting {input_file} mean/scale npy files")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_path = out_dir / (input_file.stem + "_mean.npy")
    scale_path = out_dir / (input_file.stem + "_scale.npy")
    var_path = out_dir / (input_file.stem + "_var.npy")

    np.save(mean_path, scaler.mean_, allow_pickle=False)
    np.save(scale_path, scaler.scale_, allow_pickle=False)
    np.save(var_path, scaler.var_, allow_pickle=False)
