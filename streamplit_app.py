
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="WiFi Motion Detection",
    page_icon="📡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

LABELS = {
    0: "NOT MOVING",
    1: "MOVING",
}

STATE_NAMES = {
    "empty_room": "Empty Room",
    "standing_still": "Standing Still",
    "moving": "Moving",
}


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(
        str(MODEL_DIR / "xgboost_motion.json")
    )
    return model


@st.cache_data
def load_data():
    windows = pd.read_csv(
        DATA_DIR / "processed_windows.csv"
    )

    test = pd.read_csv(
        MODEL_DIR / "test_predictions.csv"
    )

    with open(
            MODEL_DIR / "feature_columns.json"
    ) as file:
        features = json.load(file)

    return windows, test, features


required_files = [
    DATA_DIR / "processed_windows.csv",
    MODEL_DIR / "test_predictions.csv",
    MODEL_DIR / "feature_columns.json",
    MODEL_DIR / "xgboost_motion.json",
    ]

missing = [
    str(path.relative_to(ROOT))
    for path in required_files
    if not path.exists()
]

if missing:
    st.error("Missing files:")
    st.code("\n".join(missing))
    st.stop()

try:
    model = load_model()
    windows, test, FEATURES = load_data()

    # Match test predictions to their CSI features.
    # Rounding protects against tiny CSV float differences.
    keys = ["state", "start_s", "end_s"]

    for column in ["start_s", "end_s"]:
        windows[column] = windows[column].round(5)
        test[column] = test[column].round(5)

    test = test.merge(
        windows[keys + FEATURES],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if test[FEATURES].isna().any().any():
        raise ValueError(
            "Some test windows could not be matched "
            "to their CSI features."
        )

    X_test = test[FEATURES].astype(float)

    # Verify saved predictions match the saved model.
    model_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    if not np.allclose(
            model_probabilities,
            test["movement_probability"].to_numpy(),
            atol=1e-5,
    ):
        raise ValueError(
            "Model and test_predictions.csv do not "
            "match. Rerun train_model.py."
        )

    # XGBoost's built-in SHAP contributions.
    # The final column is the baseline / bias term.
    dmatrix = xgb.DMatrix(
        X_test,
        feature_names=FEATURES,
    )

    contributions = model.get_booster().predict(
        dmatrix,
        pred_contribs=True,
    )

    shap_values = contributions[:, :-1]
    bias_values = contributions[:, -1]

except Exception as error:
    st.error(f"Could not load dashboard: {error}")
    st.stop()


# ==========================================
# ADD LABELS AND ERROR FLAGS
# ==========================================

test["actual"] = (
    test["label"].astype(int).map(LABELS)
)

test["predicted"] = (
    test["predicted_label"].astype(int).map(LABELS)
)

test["correct"] = (
        test["label"].astype(int)
        == test["predicted_label"].astype(int)
)

test["result"] = np.where(
    test["correct"],
    "CORRECT",
    "INCORRECT",
)

# Largest absolute feature contribution per row
top_feature_indices = np.abs(
    shap_values
).argmax(axis=1)

test["top_feature"] = [
    FEATURES[index]
    for index in top_feature_indices
]

test["top_feature_contribution"] = [
    float(shap_values[row, index])
    for row, index in enumerate(top_feature_indices)
]

test["row_id"] = np.arange(len(test))


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Dashboard Filters")

state_filter = st.sidebar.selectbox(
    "Experiment",
    ["All"] + list(STATE_NAMES.keys()),
    format_func=lambda value: (
        "All experiments"
        if value == "All"
        else STATE_NAMES[value]
    ),
    )

result_filter = st.sidebar.radio(
    "Classification result",
    ["All", "Correct only", "Incorrect only"],
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Model:** XGBoost

    **Input:** ESP32 Wi-Fi CSI

    **Features:** 22

    **Window length:** 6 seconds

    **Classes:** MOVING / NOT MOVING
    """
)


# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("WiFi CSI — XGBoost Evaluation")

st.caption(
    "ESP32 Motion Detection | Test Predictions "
    "| Classification Errors | Feature Attribution"
)

st.warning(
    "Evaluation uses a temporal holdout from the "
    "same three recordings used for training. "
    "These results are not independent-session "
    "validation."
)


# ==========================================
# SECTION 1: STATISTICS
# ==========================================

st.header("1. Classification Statistics")

y_true = test["label"].astype(int)
y_pred = test["predicted_label"].astype(int)

total = len(test)
correct_count = int(test["correct"].sum())
incorrect_count = total - correct_count

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

macro_f1 = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0,
)

cols = st.columns(4)

cols[0].metric("Test Windows", total)
cols[1].metric("Correct", correct_count)
cols[2].metric("Incorrect", incorrect_count)
cols[3].metric("Accuracy", f"{accuracy:.2%}")

cols = st.columns(4)

cols[0].metric("Precision", f"{precision:.3f}")
cols[1].metric("Recall", f"{recall:.3f}")
cols[2].metric("F1 (Moving)", f"{f1:.3f}")
cols[3].metric("Macro F1", f"{macro_f1:.3f}")

st.divider()


# ==========================================
# SECTION 2: CONFUSION MATRIX
# ==========================================

st.header("2. Correct vs Incorrect Predictions")

left, right = st.columns([1, 1])

with left:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["NOT MOVING", "MOVING"],
            y=["NOT MOVING", "MOVING"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 24},
            colorscale="Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=370,
    )

    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:
    counts = pd.DataFrame({
        "Result": ["Correct", "Incorrect"],
        "Count": [
            correct_count,
            incorrect_count,
        ],
    })

    fig = px.pie(
        counts,
        names="Result",
        values="Count",
        title="Classification Success Rate",
        color="Result",
        color_discrete_map={
            "Correct": "#20A779",
            "Incorrect": "#E45756",
        },
        hole=0.6,
    )

    fig.update_layout(height=370)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ==========================================
# SECTION 3: PERFORMANCE PER EXPERIMENT
# ==========================================

st.header("3. Performance by Experiment")

summary = (
    test.groupby("state")
    .agg(
        total=("correct", "size"),
        correct=("correct", "sum"),
    )
    .reset_index()
)

summary["incorrect"] = (
        summary["total"] - summary["correct"]
)

summary["accuracy"] = (
        summary["correct"] / summary["total"]
)

summary["experiment"] = (
    summary["state"].map(STATE_NAMES)
)

st.dataframe(
    summary[
        [
            "experiment",
            "total",
            "correct",
            "incorrect",
            "accuracy",
        ]
    ],
    hide_index=True,
    use_container_width=True,
    column_config={
        "accuracy": st.column_config.ProgressColumn(
            "Accuracy",
            min_value=0,
            max_value=1,
            format="%.1f",
        ),
    },
)

chart_data = summary.melt(
    id_vars="experiment",
    value_vars=["correct", "incorrect"],
    var_name="result",
    value_name="windows",
)

fig = px.bar(
    chart_data,
    x="experiment",
    y="windows",
    color="result",
    barmode="stack",
    title="Correct and Incorrect Windows by Experiment",
    color_discrete_map={
        "correct": "#20A779",
        "incorrect": "#E45756",
    },
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# ==========================================
# SECTION 4: ALL TEST PREDICTIONS
# ==========================================

st.header("4. Individual Test Predictions")

filtered = test.copy()

if state_filter != "All":
    filtered = filtered[
        filtered["state"] == state_filter
        ]

if result_filter == "Correct only":
    filtered = filtered[filtered["correct"]]

elif result_filter == "Incorrect only":
    filtered = filtered[~filtered["correct"]]

filtered = filtered.reset_index(drop=True)

display_columns = [
    "state",
    "start_s",
    "end_s",
    "actual",
    "predicted",
    "result",
    "movement_probability",
    "top_feature",
]

st.write(
    f"Showing **{len(filtered)}** of "
    f"**{len(test)}** test windows."
)

st.caption(
    "Click a row to inspect its CSI features "
    "and the model's explanation."
)

if filtered.empty:
    st.info("No predictions match these filters.")

else:
    event = st.dataframe(
        filtered[display_columns],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "movement_probability":
                st.column_config.ProgressColumn(
                    "Movement Score",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
            "top_feature": "Strongest CSI Feature",
            "actual": "Ground Truth",
            "predicted": "Prediction",
        },
    )

    selected_rows = event.selection.rows

    selected_position = (
        selected_rows[0]
        if selected_rows
        else 0
    )

    selected = filtered.iloc[selected_position]

    # Original position in the full test dataset
    original_index = int(selected["row_id"])

    st.divider()

    # ======================================
    # SECTION 5: EXPLAIN ONE PREDICTION
    # ======================================

    st.header("5. Explain Selected Prediction")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Actual Label",
        selected["actual"],
    )

    col2.metric(
        "Predicted Label",
        selected["predicted"],
    )

    col3.metric(
        "Movement Score",
        f"{selected['movement_probability']:.2%}",
    )

    col4.metric(
        "Classification",
        selected["result"],
    )

    st.write(
        "**Experiment:**",
        STATE_NAMES.get(
            selected["state"],
            selected["state"],
        ),
    )

    st.write(
        "**Time window:**",
        f"{selected['start_s']:.1f}–"
        f"{selected['end_s']:.1f} seconds",
    )

    if selected["correct"]:
        st.success(
            "The model predicted the correct label."
        )
    else:
        st.error(
            "The model predicted the wrong label."
        )

    # --------------------------------------
    # Local SHAP contributions
    # --------------------------------------

    st.subheader(
        "Features That Influenced This Decision"
    )

    local_shap = shap_values[original_index]

    explanation = pd.DataFrame({
        "feature": FEATURES,
        "value": selected[FEATURES].to_numpy(
            dtype=float
        ),
        "contribution": local_shap,
    })

    explanation["absolute_contribution"] = (
        explanation["contribution"].abs()
    )

    explanation["direction"] = np.where(
        explanation["contribution"] > 0,
        "Toward MOVING",
        "Toward NOT MOVING",
        )

    explanation = explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    top_features = explanation.head(12).copy()

    fig = px.bar(
        top_features.sort_values("contribution"),
        x="contribution",
        y="feature",
        color="direction",
        orientation="h",
        title="Top 12 Feature Contributions (SHAP)",
        color_discrete_map={
            "Toward MOVING": "#E45756",
            "Toward NOT MOVING": "#268BD2",
        },
        hover_data=["value"],
    )

    fig.update_layout(
        xaxis_title="Contribution to Model Log-Odds",
        yaxis_title="CSI Feature",
        height=520,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Positive SHAP contributions push the "
        "prediction toward MOVING. Negative "
        "contributions push it toward NOT MOVING. "
        "These values are in log-odds units, "
        "not probability percentages."
    )

    st.markdown("#### All 22 CSI Feature Values")

    st.dataframe(
        explanation[
            [
                "feature",
                "value",
                "contribution",
                "direction",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander(
            "Technical explanation of the prediction"
    ):
        margin = (
                float(bias_values[original_index])
                + float(local_shap.sum())
        )

        reconstructed_probability = (
                1 / (1 + np.exp(-margin))
        )

        st.write(
            "Baseline log-odds:",
            round(
                float(bias_values[original_index]),
                4,
            ),
        )

        st.write(
            "Sum of feature contributions:",
            round(float(local_shap.sum()), 4),
        )

        st.write(
            "Reconstructed probability:",
            f"{reconstructed_probability:.4f}",
        )

        st.write(
            "Saved XGBoost probability:",
            f"{selected['movement_probability']:.4f}",
        )

        st.caption(
            "The baseline and SHAP contributions "
            "add up to the prediction's log-odds."
        )


# ==========================================
# SECTION 6: GLOBAL FEATURE IMPORTANCE
# ==========================================

st.divider()
st.header("6. Global Feature Importance")

global_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_absolute_shap": np.abs(
        shap_values
    ).mean(axis=0),
})

global_importance = (
    global_importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )
)

fig = px.bar(
    global_importance.head(15).sort_values(
        "mean_absolute_shap"
    ),
    x="mean_absolute_shap",
    y="feature",
    orientation="h",
    title="Top 15 Features Across Test Windows",
)

fig.update_layout(
    height=550,
    xaxis_title="Mean Absolute SHAP Contribution",
    yaxis_title="CSI Feature",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.caption(
    "Global importance is computed from the "
    "test windows. It describes which features "
    "influenced this model's predictions most, "
    "not causal relationships."
)


# ==========================================
# DOWNLOAD
# ==========================================

st.divider()

st.header("7. Export Evaluation Results")

export_columns = [
    "state",
    "start_s",
    "end_s",
    "actual",
    "predicted",
    "result",
    "movement_probability",
    "top_feature",
    "top_feature_contribution",
]

st.download_button(
    "Download Test Predictions",
    data=test[export_columns].to_csv(
        index=False
    ).encode("utf-8"),
    file_name="xgboost_test_evaluation.csv",
    mime="text/csv",
)

st.caption(
    "Research prototype. Detects movement patterns, "
    "not epileptic seizures."
)
