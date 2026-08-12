# Evaluation Flow

This folder is separate from the main assistant implementation. It is used to run a fixed question set through the German Administrative Assistant and judge each answer with a second LLM.

## Files

```text
evaluation_flow/
  README.md
  questions.txt
  run_evaluation.py
```

## How It Works

```text
questions.txt
  -> run one question through GermanAdminGuideAgent
  -> collect answer and internal state
  -> judge agent reviews question + answer + limited evidence
  -> save JSONL, CSV, and summary files
```

By default each question uses a fresh conversation id. That keeps answers independent and avoids previous questions influencing later ones.

## Prepare Questions

Put one question per line in:

```text
evaluation_flow/questions.txt
```

Blank lines and lines starting with `#` are ignored.

## Run A Small Test

```bash
python3 evaluation_flow/run_evaluation.py --limit 3
```

## Run All Questions

```bash
python3 evaluation_flow/run_evaluation.py
```

## Output

Each run creates a timestamped folder under:

```text
evaluation_flow/results/
```

Output files:

- `results.jsonl`: full per-question records.
- `results.csv`: compact spreadsheet-friendly results.
- `summary.json`: aggregate score and pass/fail counts.

## Judge Scores

The judge returns:

- `satisfactory`: whether the answer is good enough for the user.
- `rank`: `excellent`, `good`, `partial`, `poor`, or `fail`.
- `score`: 1 to 5.
- `relevance`: 1 to 5.
- `grounding`: 1 to 5.
- `completeness`: 1 to 5.
- `clarity`: 1 to 5.
- `safety`: 1 to 5.
- `reason`: short explanation.
- `missing_or_wrong`: concrete problems.
- `improvement`: what would make the answer better.

## Useful Options

Use another question file:

```bash
python3 evaluation_flow/run_evaluation.py --questions path/to/questions.txt
```

Use a specific output directory:

```bash
python3 evaluation_flow/run_evaluation.py --out-dir evaluation_flow/results/my_run
```

Use a shared conversation for multi-turn evaluation:

```bash
python3 evaluation_flow/run_evaluation.py --shared-conversation
```

Change judge provider/model role:

```bash
python3 evaluation_flow/run_evaluation.py --judge-provider groq --judge-role supervisor
```

## Important Note

The judge is an evaluation helper, not absolute truth. For the thesis/report, use these results as structured evidence and manually inspect a sample of strong and weak cases.
