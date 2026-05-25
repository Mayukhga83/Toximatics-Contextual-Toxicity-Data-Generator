from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_config
from src.io_utils import dataframe_to_jsonl
from src.pipeline import run_pipeline


st.set_page_config(
    page_title="Toximatics: Contextual Toxicity Data Generator",
    page_icon="🤖",
    layout="wide",
)

st.title("Toximatics: Contextual Toxicity Data Generator")
st.caption(
"Generate context-dependent utterance-context pairs with controlled polarity using OpenAI models. This is a demo of the pipeline describe in our Toximatics Paper: https://aclanthology.org/2024.sigdial-1.65/. In this version we omit N-Times Multistage for its larger inference time and reccomend to use only Two Step Multistage if Multistage is required. This respository and demo has been created by Mayukh Das (TU Braunschweig) If you use this tool please consider citing our paper!"
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Paste OpenAI API Key",
        type="password",
        help="This overrides OPENAI_API_KEY from .env file for this session.",
    )

    generator_model = st.text_input("Generator model", value="gpt-4o-mini")
    validator_model = st.text_input("Validator model", value="gpt-4o-mini")

    input_mode = st.radio("Input mode", ["Upload CSV", "Write utterance manually"])

    uploaded_file = None
    manual_utterance = ""
    utterance_column = None
    df_input = None

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.write("Detected columns:")
            st.write(list(df_input.columns))
            utterance_column = st.selectbox("Select utterance column", df_input.columns)
    else:
        manual_utterance = st.text_area(
            "Write a seed utterance",
            placeholder="Example: You are so lucky to work from home.",
            height=120,
        )

    target_polarity = st.selectbox(
        "Target polarity",
        ["toxic", "benign", "neutral", "ambiguous"],
        index=0,
    )

    generation_mode_label = st.selectbox(
        "Generation mode",
        [
            "direct",
            "single_stage",
            "multistage",
        ],
        format_func=lambda x: {
            "direct": "Direct Context Augmentation",
            "single_stage": "Single-Stage New Pair",
            "multistage": "Two-Step Multistage",
        }[x],
    )

    num_examples = st.number_input(
        "Number of examples",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )

    validate = st.checkbox("Validate with critic model", value=True)
    repair_failed = st.checkbox("Repair failed validations", value=False)

    generate_button = st.button("Generate", type="primary")


with st.expander("What the generation modes mean", expanded=False):
    st.markdown(
        """
        **Direct Context Augmentation** keeps the seed utterance fixed and generates a context.

        **Single-Stage New Pair** generates a new utterance and context inspired by the seed. Therefore this creates completely new pairs.

        **Two-Step Multistage** first creates an intermediate context, then creates a more diverse final pair.
        """
    )

if uploaded_file is not None and df_input is not None:
    st.subheader("Input preview")
    st.dataframe(df_input.head(20), use_container_width=True)

if generate_button:
    try:
        config = get_config(
            api_key=api_key.strip() or None,
            generator_model=generator_model.strip() or None,
            validator_model=validator_model.strip() or None,
        )

        if input_mode == "Upload CSV":
            if df_input is None or utterance_column is None:
                st.error("Please upload a CSV and select an utterance column.")
                st.stop()
            utterances = (
                df_input[utterance_column]
                .dropna()
                .astype(str)
                .map(str.strip)
                .tolist()
            )
            utterances = [u for u in utterances if u]
        else:
            if not manual_utterance.strip():
                st.error("Please write a seed utterance.")
                st.stop()
            utterances = [manual_utterance.strip()]

        with st.spinner("Generating examples..."):
            rows = run_pipeline(
                utterances=utterances,
                target_polarity=target_polarity,
                generation_mode=generation_mode_label,
                num_examples=int(num_examples),
                config=config,
                validate=validate,
                repair_failed=repair_failed,
                shuffle=True,
                progress=False,
            )

        result_df = pd.DataFrame(rows)
        st.session_state["result_df"] = result_df

    except Exception as exc:
        st.error(f"Generation failed: {exc}")

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]

    st.subheader("Generated examples")
    st.dataframe(result_df, use_container_width=True)

    passed = result_df["validation_passed"].fillna(False).sum() if "validation_passed" in result_df else 0
    st.metric("Validation passed", f"{passed}/{len(result_df)}")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    jsonl_bytes = dataframe_to_jsonl(result_df).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="generated_contextual_toxicity_dataset.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download JSONL",
            data=jsonl_bytes,
            file_name="generated_contextual_toxicity_dataset.jsonl",
            mime="application/jsonl",
        )

    st.subheader("Readable view")
    for i, row in result_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**Example {i + 1}**")
            st.markdown(f"**Seed:** {row['seed_utterance']}")
            st.markdown(f"**Target polarity:** `{row['target_polarity']}`")
            st.markdown(f"**Generated utterance:** {row['generated_utterance']}")
            st.markdown(f"**Generated context:** {row['generated_context']}")
            st.markdown(f"**Rationale:** {row['rationale']}")
            if row.get("validator_prediction", ""):
                st.markdown(
                    f"**Validator:** `{row['validator_prediction']}` "
                    f"confidence={row['validator_confidence']}"
                )
