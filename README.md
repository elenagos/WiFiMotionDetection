# WiFi Motion Detection

A machine learning project for detecting motion from ESP32 Wi-Fi Channel State Information (CSI). The pipeline processes recorded CSI measurements, extracts temporal features, and trains an XGBoost classifier to distinguish **MOVING** from **NOT MOVING**.

A Streamlit dashboard provides interactive evaluation, error analysis, and SHAP-based explanations of individual predictions.

## Features

- CSI validation, transmitter filtering, and per-packet amplitude normalization.
- Sliding-window feature extraction using 6-second windows with a 3-second step.
- Binary classification using 22 CSI-derived features.
- Time-based training and testing with a gap between subsets.
- Classification metrics, confusion matrices, and feature importance plots.
- Interactive prediction inspection and local and global SHAP explanations.
- CSV export of evaluation results.

## Data

The preprocessing script expects three recordings in the `data/` directory:

| Recording | Scenario | Label |
|---|---|---|
| `empty_room.csv` | No person in the room | NOT MOVING |
| `standing_still.csv` | A person standing still | NOT MOVING |
| `moving.csv` | A person moving | MOVING |

Each CSV must contain:

| Column | Description |
|---|---|
| `timestamp` | Numeric timestamp in seconds. |
| `mac` | Wi-Fi transmitter MAC address. |
| `amplitudes` | String representation of a list containing 64 CSI amplitude values. |

By default, preprocessing selects a transmitter present in all three recordings. A specific transmitter can also be selected using `--mac`.

Raw recordings and trained model artifacts are not included in the repository.

## How It Works

### 1. Preprocessing

The pipeline filters packets by transmitter, removes invalid measurements, selects active subcarriers, and normalizes each packet by its mean amplitude.

Recordings are divided into overlapping windows. Windows containing fewer than eight packets are discarded.

### 2. Feature Extraction

Each window is represented by 22 features describing:

- Temporal variation across subcarriers.
- Changes between consecutive packets.
- Amplitude range and drift between the first and last packets.
- Variation and packet-to-packet changes within six subcarrier groups.

### 3. Training and Evaluation

An XGBoost classifier predicts movement from the extracted features.

For each recording, the first 70% of windows are used for training. Two windows are skipped before the remaining test portion to separate overlapping training and test windows.

The training script reports accuracy, balanced accuracy, precision, recall, F1, and macro F1. It also saves the model, test predictions, metrics, and diagnostic plots.

### 4. Interactive Dashboard

The Streamlit application displays:

- Overall classification metrics and a confusion matrix.
- Performance for each recording scenario.
- Predictions filtered by scenario and correctness.
- Movement probabilities for individual windows.
- SHAP feature contributions for selected predictions.
- Global feature importance across test windows.
- Downloadable evaluation results.

The dashboard evaluates recorded data; it does not capture a live CSI stream.

## Repository Structure

```text
WiFiMotionDetection/
├── model /                  # Directory name includes a trailing space
│   ├── preprocessing.py
│   └── xgboostmodel.py
└── streamplit_app.py
```

The `data/` and `models/` directories are used for local recordings and generated outputs.

## Getting Started

Run the following commands from the repository root.

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn xgboost streamlit plotly
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Prepare the Recordings

Create a `data/` directory and add:

```text
data/
├── empty_room.csv
├── standing_still.csv
└── moving.csv
```

### Extract Features

The quotes preserve the trailing space in the current directory name.

```bash
python "model /preprocessing.py" --data-dir data --output data/processed_windows.csv
```

To select a transmitter explicitly:

```bash
python "model /preprocessing.py" --data-dir data --mac "AA:BB:CC:DD:EE:FF"
```

### Train the Classifier

```bash
python "model /xgboostmodel.py"
```

Generated artifacts are saved in `models/`, including the trained model, evaluation metrics, test predictions, confusion matrix, and feature importance plots.

### Launch the Dashboard

```bash
streamlit run streamplit_app.py
```

## Evaluation Scope

Evaluation uses later portions of the same three recordings used for training. It provides a preliminary assessment within those recording sessions; performance across independent sessions, rooms, people, and hardware requires further validation.

This is a research prototype for movement detection, not a medically validated system or an epileptic seizure detector.
