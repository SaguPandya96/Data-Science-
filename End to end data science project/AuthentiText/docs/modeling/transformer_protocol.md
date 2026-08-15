# Transformer candidate protocol

AuthentiText's next candidate is Google's two-layer BERT-Tiny checkpoint,
`google/bert_uncased_L-2_H-128_A-2`, pinned to Hugging Face revision
`30b0a37ccaaa32f332884b96992754e246e48c5f`. Google released the small BERT
family for resource-constrained research under Apache-2.0. The selected
checkpoint has two transformer layers, a 128-wide hidden state, two attention
heads, and 4,386,178 pretrained parameters.

This is a resource decision, not a performance claim. The candidate is small
enough to benchmark on the audited CPU workstation while still providing a
real pretrained transformer comparison. Its measured results may be worse than
the lexical baseline and will be retained if that happens.

## Prespecified stages

1. Run the preflight under Python 3.11. It verifies the sanitized MAGE training
   file by size and SHA-256 without opening the test partition.
2. Resolve and lock the CPU framework dependencies in the isolated transformer
   environment. Do not add them to the API runtime.
3. Download only the pinned checkpoint revision and record every downloaded
   file's identity.
4. Run a train-only throughput and memory probe. This probe is operational
   evidence, not candidate performance evidence.
5. Fine-tune on the complete 287,843-row sanitized training partition with seed
   1729 and a maximum sequence length of 128. A reduced training sample cannot
   be substituted and presented as the comparable candidate.
6. Use the existing disjoint validation roles for evaluation, calibration, and
   policy selection. Freeze artifacts, calibration, and thresholds before any
   test evaluation.
7. Apply the complete gate in [MODEL_SELECTION.md](../MODEL_SELECTION.md),
   including human false-machine, machine false-human, coverage, calibration,
   short-text, domain, generator, and MAGE OOD results.

The first preflight is intentionally allowed to finish as `not_ready`. That
status means no transformer was trained and no performance metrics exist.

## Remote environment resolution

The dedicated transformer workflow uses Python 3.11 and reviewed direct pins
from `requirements/transformer.in`. The isolated environment includes the
existing evaluation/calibration stack so the transformer is scored under the
same validation policy as the baseline. PyTorch is pinned to the official CPU
wheel so the hosted CPU runner does not resolve an unused CUDA stack. The job
performs no training and reads no dataset. It runs `pip check`, captures the
complete resolved environment with `pip freeze --all`, records the four
framework versions, and uploads those files as a short-lived GitHub Actions
artifact. The resolved lock must be reviewed and committed before the
throughput probe or full training is allowed to run.

The CPU-only resolution completed successfully in GitHub Actions on 2026-08-15.
The exact transitive environment is committed as
[`requirements/transformer.lock`](../../requirements/transformer.lock), and its
runner, framework, workflow, and artifact identities are recorded in
[`transformer_environment_report.json`](../../data/metadata/transformer_environment_report.json).

The remote probe acquires only the pinned 403,744,528-byte MAGE training file.
It reproduces the canonical cleaned train file, applies the 69 text-free record
exclusions in
[`transformer_train_decisions.json`](../../data/metadata/transformer_train_decisions.json),
and requires the final 287,843-row decompressed canonical record stream to
match its committed SHA-256. The original gzip byte identities remain recorded
as provenance, but are not used as a cross-platform equality check because
zlib output can differ by runtime. The workflow never downloads or opens the
test partition.

The throughput probe is fixed at seed 1729, sequence length 128, batch size 32,
4 warm-up optimizer steps, and 60 measured optimizer steps over 2,048 training
rows. It reports runtime and memory only. It does not report loss, accuracy, or
any other candidate-performance metric, and it does not save the partially
updated probe model. The viability estimate reserves 15 minutes for setup and
requires a prespecified three-epoch full run to fit inside the six-hour hosted
runner limit.

The measured remote probe processed 1,920 optimizer-step rows in 12.086 seconds
(158.858 rows/second) with 776,138,752 bytes peak RSS. It estimated 1,811.950
seconds per full epoch and 6,335.850 seconds for three epochs plus reserved
setup time, so the full run cleared the resource gate. These are operational
measurements, not candidate-quality metrics.

The full workflow separately rebuilds the validation role from the pinned raw
validation file and the 69 text-free audited exclusions in
[`transformer_validation_decisions.json`](../../data/metadata/transformer_validation_decisions.json).
It trains for exactly three epochs without early stopping, then scores
validation, fits the existing three-role calibration and abstention policy,
reload-checks the saved model, and uploads model and text-free evidence. The
test partition remains unavailable throughout.

The environment-resolution job is bounded to 30 minutes. The later full
training job must stay within GitHub's six-hour hosted-runner limit; a measured
throughput probe will decide whether that target is viable rather than assuming
that BERT-Tiny will finish in time.

## Current workstation result

The 2026-08-15 preflight found a supported Python 3.11 interpreter, eight
logical CPUs, sufficient memory, sufficient free disk, and the verified full
training partition. PyTorch, Transformers, Tokenizers, and Accelerate are not
installed, and the pinned BERT-Tiny revision is not cached. Package-index and
model downloads are blocked in the current workspace, so training cannot start
here without network access.

Sources: [Google's small BERT release](https://github.com/google-research/bert),
[pinned model revision](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f),
and [PyTorch's Windows installation requirements](https://docs.pytorch.org/get-started/locally/).
