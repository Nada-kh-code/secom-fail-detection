"""
SECOM Fail Detector — Streamlit app
Upload production runs (590 sensor columns) → fail-risk score + the sensors behind it.
Run locally:  streamlit run app.py
"""
from pathlib import Path
import re
from io import StringIO
import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Importing from secom_pipeline also lets joblib find the custom transformers
from secom_pipeline import SENSOR_COLS, validate_upload

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "secom_fail_detector.joblib"
SAMPLE_PATH = ROOT / "sample_data" / "sample_runs.csv"

st.set_page_config(page_title="SECOM Fail Detector", page_icon="🔬", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load the model file ({MODEL_PATH.name}). "
             f"Run the notebook first to create it.\n\n{e}")
    st.stop()

pipe = bundle["pipeline"]
m = bundle["test_metrics"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Decision threshold")
    threshold = st.slider(
        "Flag a run as **Fail risk** when its risk score ≥",
        min_value=0.0, max_value=1.0,
        value=float(round(bundle["threshold"], 3)), step=0.001, format="%.3f",
        help="Lower = catch more failing runs but inspect more good runs. "
             "The score ranks runs by risk; it is not a calibrated probability.")
    st.caption(f"Recommended (max F2 on training data): **{bundle['threshold']:.3f}**")

    st.divider()
    st.header("📊 Model")
    st.write(f"**{bundle['model_name']}** on {len(bundle['selected_sensors'])} selected sensors")
    c1, c2 = st.columns(2)
    c1.metric("Test PR-AUC", f"{m['PR-AUC']:.2f}",
              help=f"Baseline = fail rate ≈ {bundle['train_fail_rate']:.2f}")
    c2.metric("Test ROC-AUC", f"{m['ROC-AUC']:.2f}")
    c1.metric("Recall (Fail)", f"{m['Recall (Fail)']:.0%}")
    c2.metric("Precision (Fail)", f"{m['Precision (Fail)']:.0%}")
    st.caption("Data: SECOM, UCI ML Repository (McCann & Johnston, 2008).")

# ---------------- Header ----------------
st.title("🔬 Semiconductor Run Fail Detector")
st.write("Upload the sensor readings of one or more production runs. The model estimates the risk "
         "that each run **fails the line test** and shows which sensors drive that risk.")
st.info("Decision support only — it helps decide which runs to inspect early; "
        "it does not replace the line test.", icon="ℹ️")
# ---------------- Helpers ----------------
MAX_ROWS = 5000


def _is_number(x):
    """True if a column name looks like a number (so the first row was data, not a header)."""
    s = str(x)
    if s.lower().startswith("nan"):
        s = re.sub(r"\.\d+$", "", s)       # pandas renames duplicate 'nan' headers to 'nan.1', 'nan.2' ...
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_any_table(uploaded):
    """Read a CSV with a header, or a raw whitespace-separated file like secom.data."""
    text = uploaded.getvalue().decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("The file is empty.")
    df = pd.read_csv(StringIO(text), sep=None, engine="python")
    if df.shape[1] == 1:                                         # whitespace-separated, no header
        df = pd.read_csv(StringIO(text), sep=r"\s+", header=None)
    elif np.mean([_is_number(c) for c in df.columns]) > 0.9:     # first row was data, not a header
        df = pd.read_csv(StringIO(text), sep=None, engine="python", header=None)
    return df


# ---------------- Tabs ----------------
tab_pred, tab_sensors, tab_help = st.tabs(["Predict runs", "Key sensors (global)", "File format & help"])
result = None   # filled when a file is scored successfully (used in part 3)

with tab_pred:
    col_u, col_s = st.columns([3, 1])
    uploaded = col_u.file_uploader("Production-run file (.csv, .txt or .data)",
                                   type=["csv", "txt", "data"])
    if col_s.button("Try the sample file", width="stretch", disabled=not SAMPLE_PATH.exists()):
        st.session_state["use_sample"] = True
    if uploaded is not None:
        st.session_state["use_sample"] = False

    raw = None
    try:
        if uploaded is not None:
            raw = read_any_table(uploaded)
        elif st.session_state.get("use_sample"):
            raw = pd.read_csv(SAMPLE_PATH)
            st.caption(f"Using `{SAMPLE_PATH.name}` — 25 held-out test runs.")
    except Exception as e:
        st.error(f"Could not read the file: {e}")

    if raw is None:
        st.write("⬆️ Upload a file or click **Try the sample file**.")
    else:
        try:
            if len(raw) > MAX_ROWS:
                st.warning(f"Only the first {MAX_ROWS:,} rows are scored.")
                raw = raw.head(MAX_ROWS)

            X, notes = validate_upload(raw)
            for n in notes:
                st.warning(n)

            ids = (raw["run_id"].astype(str).to_numpy() if "run_id" in raw.columns
                   else np.array([f"row {i + 1}" for i in range(len(raw))]))
            proba = pipe.predict_proba(X)[:, 1]
            flag = proba >= threshold

            k1, k2, k3 = st.columns(3)
            k1.metric("Runs scored", len(X))
            k2.metric("Flagged as Fail risk", int(flag.sum()))
            k3.metric("Share flagged", f"{flag.mean():.0%}")

            res = pd.DataFrame({"run_id": ids,
                                "risk score": proba.round(3),
                                "decision": np.where(flag, "⚠️ Fail risk — inspect", "✅ OK")})
            if "label" in raw.columns:
                res["actual (from file)"] = np.where(raw["label"].to_numpy() == 1, "Fail", "Pass")
            res = res.sort_values("risk score", ascending=False).reset_index(drop=True)

            st.dataframe(res, width="stretch", hide_index=True,
                         column_config={"risk score": st.column_config.ProgressColumn(
                             "risk score", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("Download predictions (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "secom_predictions.csv", "text/csv")
            result = (ids, X, proba, flag)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong while scoring the file: {e}")
            
# ---------------- Part 3: explanations ----------------
selected = tuple(bundle["selected_sensors"])
baseline = bundle["baseline_values"]


@st.cache_data(show_spinner=False)
def explain_run(_pipe, row):
    """Reset one sensor at a time to its typical (training-median) value and measure
    how much the risk score changes. Positive = this sensor's value raises the risk."""
    p0 = _pipe.predict_proba(row)[:, 1][0]
    batch = pd.concat([row] * len(selected), ignore_index=True)
    for i, s in enumerate(selected):
        batch.loc[i, s] = baseline[s]
    p = _pipe.predict_proba(batch)[:, 1]
    return pd.DataFrame({
        "sensor": selected,
        "value": row[list(selected)].iloc[0].to_numpy(),
        "typical (train median)": [baseline[s] for s in selected],
        "effect on risk": p0 - p,
    }).sort_values("effect on risk", ascending=False)


if result is not None:
    ids, X, proba, flag = result
    with tab_pred:
        st.subheader("Why this run?")
        order = np.argsort(-proba)
        run = st.selectbox("Pick a run to explain (sorted by risk)", ids[order].tolist())
        pos = int(np.where(ids == run)[0][0])
        row = X.iloc[[pos]].reset_index(drop=True)
        expl = explain_run(pipe, row)

        top = pd.concat([expl.head(5), expl.tail(5)]).drop_duplicates("sensor")
        top = top[top["effect on risk"].abs() > 1e-6]
        st.write(f"Risk score for **{run}** = **{proba[pos]:.3f}** "
                 f"({'above' if flag[pos] else 'below'} the threshold {threshold:.3f}).")
        if top.empty:
            st.write("No single sensor moves this run's risk noticeably.")
        else:
            top = top.assign(direction=np.where(top["effect on risk"] > 0, "raises risk", "lowers risk"))
            chart = alt.Chart(top).mark_bar().encode(
                x=alt.X("effect on risk:Q", title="change in risk score vs. a typical value"),
                y=alt.Y("sensor:N", sort="-x", title=None),
                color=alt.Color("direction:N", scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=["#d62728", "#1f77b4"])),
                tooltip=["sensor",
                         alt.Tooltip("value:Q", format=".4f"),
                         alt.Tooltip("typical (train median):Q", format=".4f"),
                         alt.Tooltip("effect on risk:Q", format=".3f")],
            ).properties(height=320)
            st.altair_chart(chart, width="stretch")
            st.caption("Each bar: how much the risk score would drop (red) or rise (blue) if that sensor "
                       "were set to its typical training value, all other sensors unchanged.")

# ---------------- Key sensors tab ----------------
with tab_sensors:
    imp = pd.DataFrame(bundle["sensor_importance"]).head(10)
    st.write("The 10 sensors whose shuffling hurts the model most on unseen test runs "
             "(permutation importance = drop in PR-AUC). Start root-cause reviews here.")
    chart = alt.Chart(imp).mark_bar(color="#d62728").encode(
        x=alt.X("importance:Q", title="drop in test PR-AUC when shuffled"),
        y=alt.Y("sensor:N", sort="-x", title=None),
        tooltip=["sensor", alt.Tooltip("importance:Q", format=".4f"),
                 alt.Tooltip("std:Q", format=".4f")])
    st.altair_chart(chart.properties(height=330), width="stretch")
    st.caption("Importance shows association with failure, not proven cause.")

# ---------------- Help tab ----------------
with tab_help:
    st.markdown(f"""
**Accepted files**
- A CSV with columns `sensor_001` … `sensor_590` (optional `run_id` and `label` columns).
- Or a raw file with exactly 590 numeric columns and no header (same layout as `secom.data`).

**Missing values** are allowed (leave the cell empty or write `NaN`); they are filled with typical
training values. Text in a sensor column is treated as missing.

**Threshold** — the slider in the sidebar sets how cautious the detector is. The recommended value
({bundle['threshold']:.3f}) maximises F2 (recall counted twice as much as precision) on training data.
Lower it to catch more failing runs; raise it to inspect fewer good runs.
""")
    template = pd.DataFrame(columns=["run_id"] + SENSOR_COLS)
    st.download_button("Download an empty template (CSV)", template.to_csv(index=False).encode("utf-8"),
                       "secom_template.csv", "text/csv")