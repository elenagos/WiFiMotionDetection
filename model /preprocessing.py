
import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
WINDOW_SEC = 6.0
HOP_SEC = 3.0
MIN_PACKETS = 8

# Labels for binary classification
STATES = {
    "empty_room": 0,
    "standing_still": 0,
    "moving": 1,
}

# Active CSI subcarriers for our recordings
# Positions 0 and 27-37 are excluded.
ACTIVE = np.array([
    i for i in range(64)
    if i != 0 and not (27 <= i <= 37)
])


def select_common_sender(paths):
    """Find a common Wi-Fi transmitter across recordings."""

    counts = []

    for path in paths:
        df = pd.read_csv(path, usecols=["mac"])
        mac_counts = Counter(
            df["mac"].astype(str).str.upper()
        )
        counts.append(mac_counts)

    common = set(counts[0])

    for count in counts[1:]:
        common &= set(count)

    if not common:
        raise ValueError(
            "No common transmitter MAC found."
        )

    return max(
        sorted(common),
        key=lambda mac: min(c[mac] for c in counts)
    )


def parse_amplitudes(value):
    """Convert CSI amplitude text into a NumPy array."""

    try:
        values = ast.literal_eval(value)
        array = np.asarray(values, dtype=np.float32)

        if array.shape != (64,):
            return None

        if not np.isfinite(array).all():
            return None

        return array

    except (ValueError, SyntaxError, TypeError):
        return None


def load_csi_data(file_path, target_mac):
    """Load, filter and normalize CSI measurements."""

    df = pd.read_csv(file_path)

    required = {"timestamp", "mac", "amplitudes"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {file_path}"
        )

    # Keep only packets from the selected transmitter
    df = df[
        df["mac"].astype(str).str.upper()
        == target_mac.upper()
        ].copy()

    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
    )

    amplitudes = []
    valid_indices = []

    for idx, value in df["amplitudes"].items():

        array = parse_amplitudes(value)

        if array is not None:
            amplitudes.append(array)
            valid_indices.append(idx)

    if not amplitudes:
        raise ValueError(
            f"No valid CSI packets in {file_path}"
        )

    df = df.loc[valid_indices].reset_index(drop=True)

    timestamps = df["timestamp"].to_numpy(
        dtype=np.float64
    )

    # Relative time starts at 0 seconds
    timestamps = timestamps - timestamps[0]

    # Convert packets into matrix:
    # number_of_packets x number_of_subcarriers
    csi = np.stack(amplitudes)[:, ACTIVE]

    # Normalize each packet by its mean amplitude
    scale = np.maximum(
        csi.mean(axis=1, keepdims=True),
        1e-5
    )

    csi = csi / scale

    return timestamps, csi


def extract_features(csi_window):
    """Calculate motion-related features for one window."""

    temporal_std = np.std(csi_window, axis=0)

    # Changes between consecutive packets
    delta = np.abs(
        np.diff(csi_window, axis=0)
    )

    row_changes = delta.mean(axis=1)

    features = {
        "temporal_std_mean": float(temporal_std.mean()),
        "temporal_std_median": float(np.median(temporal_std)),
        "temporal_std_p90": float(np.percentile(temporal_std, 90)),
        "temporal_std_max": float(temporal_std.max()),
        "delta_mean": float(delta.mean()),
        "delta_p90": float(np.percentile(delta, 90)),
        "delta_std": float(delta.std()),
        "rowchange_std": float(row_changes.std()),
        "profile_drift": float(
            np.abs(csi_window[-1] - csi_window[0]).mean()
        ),
        "range_mean": float(
            np.ptp(csi_window, axis=0).mean()
        ),
    }

    # Divide subcarriers into six frequency groups
    groups = np.array_split(
        np.arange(csi_window.shape[1]), 6
    )

    for i, group in enumerate(groups, start=1):

        part = csi_window[:, group]

        features[f"bin{i}_std"] = float(
            np.std(part, axis=0).mean()
        )

        features[f"bin{i}_delta"] = float(
            np.abs(np.diff(part, axis=0)).mean()
        )

    return features


def create_windows(timestamps, csi):
    """Convert a recording into fixed-length windows."""

    results = []

    duration = float(timestamps[-1])

    for start in np.arange(
            0, duration - WINDOW_SEC + 1e-6, HOP_SEC
    ):

        end = start + WINDOW_SEC

        mask = (
                (timestamps >= start)
                & (timestamps < end)
        )

        window = csi[mask]

        if len(window) < MIN_PACKETS:
            continue

        features = extract_features(window)

        features["start_s"] = float(start)
        features["end_s"] = float(end)
        features["n_packets"] = int(len(window))

        results.append(features)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed_windows.csv")
    )

    parser.add_argument(
        "--mac",
        default=None,
        help="Specific transmitter MAC address"
    )

    args = parser.parse_args()

    paths = [
        args.data_dir / f"{state}.csv"
        for state in STATES
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    target_mac = (
        args.mac
        if args.mac
        else select_common_sender(paths)
    )

    all_windows = []

    for state, label in STATES.items():

        file_path = args.data_dir / f"{state}.csv"

        timestamps, csi = load_csi_data(
            file_path, target_mac
        )

        windows = create_windows(timestamps, csi)

        if windows.empty:
            raise ValueError(
                f"No valid windows for {state}"
            )

        windows["state"] = state
        windows["label"] = label
        windows["duration_s"] = float(timestamps[-1])

        all_windows.append(windows)

        print(
            f"{state}: {len(windows)} windows, "
            f"{len(csi)} packets"
        )

    dataset = pd.concat(
        all_windows,
        ignore_index=True
    )

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    dataset.to_csv(
        args.output,
        index=False
    )

    print(f"\nSelected MAC: {target_mac}")
    print(f"Total windows: {len(dataset)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
