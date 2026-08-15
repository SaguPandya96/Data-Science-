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
