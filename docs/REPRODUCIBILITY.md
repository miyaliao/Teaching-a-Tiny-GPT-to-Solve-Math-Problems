# Reproducibility and repository scope

This repository is a **lightweight source-and-results showcase**, not a self-contained copy of the course submission ZIP.

## Included

- `dpo/dpo.ipynb`: full training/evaluation **source notebook**, saved without large execution outputs.
- `model.py`: course-supplied NanoGPT architecture.
- `sft/meta.pkl`: small course-supplied character-level tokenizer metadata (74 characters).
- `scripts/`: historical preference-data builder, math-fact parsing, training, and evaluation dependencies.
- `requirements.txt`, documentation, and verified metrics.

## External artifacts deliberately omitted

- `sft/gpt.pt`: original pretrained model supplied by the course.
- `dpo/dpo.pt`: the selected Round9 Random E5 fine-tuned checkpoint, retained in the separate submission archive.
- `dpo/pos_neg_pairs.json`: selected Random100k preference pairs, retained in the submission archive.
- Large frozen Round3/4/5/7/9 stage data and benchmark files/manifests, plus historical checkpoint lineage.

**The GitHub checkout alone cannot run the full 13-epoch historical replay.** Source scripts are included for inspection and reuse, but the large input data is not.

## Setup for a new course-mode experiment

Use Python 3.10+ and install the appropriate PyTorch build for your CUDA environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the course-provided checkpoint `sft/gpt.pt` and place it alongside `sft/meta.pkl`. Open `dpo/dpo.ipynb` with the working directory set to the repository root or `dpo/`. The notebook defaults to `TRAINING_POLICY = "auto"`:

- **historical:** selected only when frozen stage data and manifests are available and the archived Random100k SHA256 matches. This is the seven-stage, 13-epoch historical route.
- **course:** used otherwise. It loads `dpo/pos_neg_pairs.json` when present or synthesizes a **different** valid 100k preference dataset, then runs five SFT→DPO epochs. This is not historical E5 reproduction.

For quick checks, `FAST_SMOKE=True` shortens training. Training creates new checkpoints; work in a disposable copy if you want to preserve previously selected weights.

## Saved experiment outputs

The executed notebook from the course submission and the final `dpo.pt` are held separately. The version tracked here is a **source-code notebook without saved cell outputs**. Historical selected E5 metrics were Val768 614/768 and Teacher680 521/680; these are development results, not TA private evaluation. An independent replay gave similar aggregate metrics but not bitwise-equal weights.

## Attribution and distribution

The NanoGPT model architecture, pretrained checkpoint, tokenizer metadata, and starter notebook scaffold were supplied as part of NTU SC3000/CZ3005. Review course redistribution conditions before making this repository public.
