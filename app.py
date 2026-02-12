import streamlit as st
import pandas as pd
from datasanity import check_dataset

st.set_page_config(page_title="DataSanity", layout="wide")

with st.sidebar:
    st.title("DataSanity 🧠")
    st.caption("Dataset health + ML strategy assistant for tabular ML.")
    st.markdown("**How to use**")
    st.markdown("- Upload a CSV\n- Select the target column\n- Run checks\n- Download the HTML report")

from pathlib import Path

def load_css(path: str):
    p = Path(path)
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css("assets/styles/streamlit.css")
st.title("🧠 DataSanity — Dataset Health Check")
st.caption("Upload a CSV and detect common ML dataset issues before training.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head(30), use_container_width=True)

target = st.selectbox("Select target column", df.columns)

if st.button("Run check", type="primary"):
    report = check_dataset(df, target)
    r = report.results

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows", r["shape"][0])
    with c2:
        st.metric("Columns", r["shape"][1])

    st.divider()
    st.subheader("📊 Dataset health score")

    r = r or {}
    s = r.get("severity") or {}

    score = s.get("score", 0)
    level = s.get("risk_level", "n/a")
    color = s.get("color", "green")

    color_map = {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴"
    }

    icon = color_map.get(color, "⚪")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Score", f"{score}/100")
    with c2:
        st.metric("Risk level", f"{icon} {level}")

    reasons = s.get("reasons", [])
    if reasons:
        st.warning("**Risk factors:**\n- " + "\n- ".join(reasons))
    else:
        st.success("No major risks detected.")


    # Warnings + sections
    st.subheader("⚖️ Target analysis")
    im = r["imbalance"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Task hint", im.get("task_hint", "n/a"))
    with c2:
        st.metric("Unique values", im.get("n_unique", "n/a"))
    with c3:
        st.metric("Numeric", str(im.get("is_numeric", "n/a")))

    if im.get("warning"):
        st.warning(im["warning"])
    if im.get("recommendation"):
        st.info(im["recommendation"])

    with st.expander("Show target distribution"):
        st.write(im.get("distribution", {}))

    st.subheader("🧭 Modeling advice")
    a = r.get("advice", {})

    st.write(f"**Task:** {a.get('task_hint', 'n/a')}")

    top_risks = a.get("top_risks", [])
    if top_risks:
        st.warning("**Top risks:**\n- " + "\n- ".join(top_risks))
    else:
        st.success("No major risks detected.")

    actions = a.get("recommended_actions", [])
    if actions:
        st.info("**Recommended actions:**\n- " + "\n- ".join(actions))

    st.subheader("❗ Missing values (>30%)")
    if len(r["missing"]["high_missing_columns"]) == 0:
        st.success("No columns above 30% missing.")
    else:
        st.write(r["missing"]["high_missing_columns"])
        if r["missing"].get("warning"):
            st.warning(r["missing"]["warning"])

    st.subheader("🧱 Constant columns")
    if len(r["constants"]["constant_columns"]) == 0:
        st.success("No constant columns.")
    else:
        st.write(r["constants"]["constant_columns"])
        if r["constants"].get("warning"):
            st.warning(r["constants"]["warning"])

    st.subheader("🆔 ID-like columns")
    if len(r["id_columns"]["id_like_columns"]) == 0:
        st.success("No ID-like columns.")
    else:
        st.write(r["id_columns"]["id_like_columns"])
        if r["id_columns"].get("warning"):
            st.warning(r["id_columns"]["warning"])

    st.subheader("🚨 Possible target leakage (corr > 0.95, numeric only)")
    if len(r["leakage"]["suspicious_features"]) == 0:
        st.success("No suspicious correlations found.")
    else:
        st.error("Suspicious correlation with target detected.")
        st.write(r["leakage"]["suspicious_features"])
        if r["leakage"].get("warning"):
            st.error(r["leakage"]["warning"])

    st.subheader("🔁 Duplicate rows")
    if r["duplicates"]["num_duplicates"] == 0:
        st.success("No duplicate rows.")
    else:
        st.warning(f"Found {r['duplicates']['num_duplicates']} duplicate rows.")
    st.subheader("🤖 Model suggestions")

    ms = (r or {}).get("model_suggestion") or {}
    mix = ms.get("feature_mix", {}) or {}

    # Header pills
    colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
    with colA:
        st.markdown(f"**Task:** `{ms.get('task_hint','n/a')}`")
    with colB:
        st.markdown(f"**Rows:** `{ms.get('n_rows','?')}`")
    with colC:
        st.markdown(f"**Features:** `{ms.get('n_features','?')}`")
    with colD:
        st.markdown(
            f"**Cat/Numeric:** `{mix.get('n_categorical','?')}/{mix.get('n_numeric','?')}`"
        )

    st.markdown("### Top recommended models")

    top_models = ms.get("top_models", []) or []
    if not top_models:
        st.info("No model suggestions available.")
    else:
        for i, m in enumerate(top_models, start=1):
            with st.container(border=True):
                st.markdown(f"#### #{i} — {m.get('model','')}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Why this model?**")
                    why = m.get("why", []) or []
                    if why:
                        st.markdown("\n".join([f"- {x}" for x in why]))
                    else:
                        st.caption("—")
                with c2:
                    st.markdown("**Best used when**")
                    when = m.get("when_to_use", []) or []
                    if when:
                        st.markdown("\n".join([f"- {x}" for x in when]))
                    else:
                        st.caption("—")

                notes = m.get("notes", []) or []
                if notes:
                    st.markdown("**Practical notes**")
                    st.markdown("\n".join([f"- {x}" for x in notes]))

    st.markdown("### Suggested baseline workflow")
    bp = ms.get("baseline_plan", {}) or {}
    if bp:
        with st.container(border=True):
            st.markdown(f"**Split:** {bp.get('split','')}")
            st.markdown(f"**Metrics:** {bp.get('metrics','')}")
            st.markdown("**Pipeline steps:**")
            st.markdown("\n".join([f"- {x}" for x in (bp.get('pipeline', []) or [])]))
    else:
        st.info("No baseline workflow available.")

    st.markdown("### 🧩 Ready-to-run training code")
    code = (r or {}).get("code_snippet", "")
    if code:
        st.code(code, language="python")
    else:
        st.info("No code snippet available.")

    # Download HTML report
    st.divider()
    html = report.to_html()
    st.download_button(
        label="⬇️ Download HTML report",
        data=html,
        file_name="datasanity_report.html",
        mime="text/html",
    )
