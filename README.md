# Contextual Toxicity Data Generator

A Streamlit + CLI demo for generating context-dependent utterance-context pairs with controlled polarity.

The idea is inspired by the research paper Toximatics: Towards Understanding Toxicity in Real-Life Social Situations (https://aclanthology.org/2024.sigdial-1.65/ and https://github.com/Mayukhga83/Toximatics) 
This is a contextual toxicity dataset-generation pipeline: instead of judging toxicity from an utterance alone, the demo generates a social context in which the utterance becomes toxic, benign, neutral, or ambiguous.

## Features

- Add OpenAI API key from Streamlit sidebar or `.env`
- Upload CSV or write one manual utterance
- Select utterance column
- Select target polarity
- Select generation mode
- Choose number of examples
- Validate generated examples with a critic model
- Optional repair step for failed generations
- Download CSV or JSONL
- Run the same pipeline from the command line

## Generation modes

### `direct`

Keeps the seed utterance fixed and generates a context.

Example:

```text
Seed: You are so lucky to work from home.
Target: toxic
```

Output: same utterance, but a generated context where it sounds dismissive or harmful.

### `single_stage`

Generates a new utterance and a new context inspired by the seed.

### `multistage`

Creates an intermediate context first, then creates a more diverse final utterance-context pair.

## Setup

```bash
cd context_toxicity_generator
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_VALIDATOR_MODEL=gpt-4o-mini
```

The Streamlit sidebar key overrides `.env`.

## Run Streamlit

```bash
streamlit run app.py
```

Then use the sidebar:

1. Paste OpenAI API key
2. Upload `data/sample_seed_utterances.csv` or write one utterance
3. Select utterance column, for example `utterance`
4. Select target polarity
5. Select generation mode
6. Choose number of examples
7. Click **Generate**
8. Download CSV or JSONL

## CLI usage

Show help:

```bash
python cli.py --help
```

Generate from CSV:

```bash
python cli.py ^
  --input data/sample_seed_utterances.csv ^
  --utterance-column utterance ^
  --target-polarity toxic ^
  --generation-mode direct ^
  --num-examples 10 ^
  --output data/outputs/generated_toxic.csv
```

Mac/Linux version:

```bash
python cli.py \
  --input data/sample_seed_utterances.csv \
  --utterance-column utterance \
  --target-polarity toxic \
  --generation-mode direct \
  --num-examples 10 \
  --output data/outputs/generated_toxic.csv
```

Generate JSONL:

```bash
python cli.py \
  --input data/sample_seed_utterances.csv \
  --utterance-column utterance \
  --target-polarity benign \
  --generation-mode single_stage \
  --num-examples 20 \
  --output data/outputs/generated_benign.jsonl \
  --format jsonl
```

Generate from one manual utterance:

```bash
python cli.py \
  --utterance "You are so lucky to work from home." \
  --target-polarity toxic \
  --generation-mode single_stage \
  --num-examples 5 \
  --output data/outputs/manual_examples.csv
```

Skip validation:

```bash
python cli.py \
  --input data/sample_seed_utterances.csv \
  --utterance-column utterance \
  --target-polarity toxic \
  --generation-mode direct \
  --num-examples 10 \
  --no-validate
```

Repair failed examples:

```bash
python cli.py \
  --input data/sample_seed_utterances.csv \
  --utterance-column utterance \
  --target-polarity toxic \
  --generation-mode direct \
  --num-examples 10 \
  --repair-failed
```

## Output columns

- `seed_utterance`
- `target_polarity`
- `generation_mode`
- `generated_utterance`
- `generated_context`
- `rationale`
- `validator_prediction`
- `validator_confidence`
- `validation_passed`
- `validator_reason`
- `safety_ok`
- `context_dependent`
- `contains_slur`
- `too_explicit`

## Notes

- Start with small batches because each example can use two API calls: one generation call and one validation call.
- `multistage` can use more calls because it generates an intermediate example.
- For the strongest demo, use the same seed utterance with both `benign` and `toxic` target polarity and compare contexts.

### Citation
Please use the following to cite this work:
```
@inproceedings{das2024toximatics,
  title={Toximatics: Towards Understanding Toxicity in Real-Life Social Situations},
  author={Mayukh Das and Wolf-Tilo Balke},
  booktitle={Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL)},
  year={2024},
