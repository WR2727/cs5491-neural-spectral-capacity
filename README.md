# Residual Context in Neural Spectral Capacity

CS5491 group G032 milestone. This package reproduces the synthetic mechanism checks, the coefficient sweep, and the development ranking and selection tables. Architecture scoring uses only the configuration. Perplexities enter the evaluation and coefficient selection.

## Reproduce the milestone

Use Python 3.9 or later on a CPU. The Python standard library is sufficient, as recorded in `requirements.txt`. The course development data are bundled, and the commands make no network requests.

From the extracted `nsc-milestone-code` directory, run:

```sh
python -B run_milestone.py --output development_results.json
```

Expect the starter and mechanism checks to print PASS, followed by `Capacity alpha: 2; independent position alpha: 2.` The JSON contains synthetic scores, both controls' full development grids, ranking and tie counts, all eight budget/k settings, and timing diagnostics. The bundled JSON is a reference from a Windows 11 run with Python 3.13.5 and an Intel Core Ultra 9 275HX. Allow roughly two seconds on that machine. Timings will vary. Run without Python's `-O` flag so the numerical assertions are active.

To run just the starter checks or the evaluator with the selected configuration:

```sh
python -B check_starter.py --development-only
python -B evaluate.py --split development --output selected_alpha_results.json
```

The second command should report `rho=0.972745` and `controlled_tau=0.611111` for `student`, using 100 architectures and eight cap/k settings. For the complete milestone tables and the synthetic checks, use `run_milestone.py`.

## Method and protocol

- `nsc.py` uses the starter's MP quadrature, fixed initialization standard deviation of 0.02 and matrix grouping. It exposes the existing layer sum as separate attention and FFN contributions for reuse.
- `student_score.py` divides each FFN contribution by `1 + alpha * r`. The capacity method uses `r = preceding layers' original NSC / whole original NSC`. The matched position control uses `r = (layer - 1) / depth` with one-based layers. Both keep attention terms at full weight. The ablation sets `alpha = 0`. The starter's increasing whole-layer weighting is a separate teaching baseline.
- `run_milestone.py` tests the one-layer and zero-prefix limits, fixed-resource FFN reordering, later-layer-only widening, exact ablation recovery and finite positive scores. It verifies the starter development reference and checks the selected default against the sweep.
- The fixed grid is `0, 0.25, 0.5, 1, 2`. Maximize development Spearman rho against negative validation PPL and break an exact tie in favor of the smaller alpha. For the primary comparison, apply the capacity method's chosen alpha to both methods. Independently maximizing development Spearman for the position control also yields 2.
- `evaluate.py` preserves the starter's size control, 12-significant-digit score comparisons and selection protocol. Use equal hidden width and `min(total)/max(total) >= 0.90` for the restricted pair statistic. Selection caps are 10M, 20M, 30M and 50M archived total parameters, with k = 1 and 3. The 200 trials per setting use seed `202627 + parameter_cap + k`. The standard deviation is the population SD across tie-breaking or uniform random choices. The archive offers one endpoint per architecture, so training-seed variability is unavailable.
- Reference FLOPs are twice the sum of scored projection weights and dense QK/AV MACs per query token at key length 384. They omit embeddings, vocabulary projection, normalization, bias, softmax, activations and cache preparation. They are an operation-count reference. The selection budget uses the archive's `total` field, including parameters outside the scored matrices.

## Data and final holdout

`data/provenance.json` records the supplied panel's version, source, file hashes and scope. The course archive attributes its 200 GPT-2/Transformer-XL configurations to [Javaheripi et al., LiteTransformerSearch (NeurIPS 2022)](https://arxiv.org/abs/2203.02094) and identifies its use in [Zhu, Zhao and Lu, Neural Spectral Capacity (2026)](https://arxiv.org/abs/2609.23087). The bundled data retain the official architecture IDs, resource counts, fixed 100/100 split and 100 development validation PPLs. They contain adaptive embeddings, memory length 192 and post-normalization. No training or data preparation is needed for the milestone.

The milestone package omits final outcome labels. For the end-of-project evaluation, freeze the score, coefficients and code in Git before copying the original starter's `data/final_labels.json` into this package's `data/`. Then verify its published hash and evaluate the 100 final architectures using their archived test PPL:

```sh
python -B check_starter.py
python -B evaluate.py --split final --output final_results.json
```

The local course project instructions call for a public Git URL with a commit hash or tag in Canvas. This folder is ready to review and commit as a standalone milestone code package.
