# Ghostbuster qualitative error-review protocol

This review examines actual frozen-policy outcomes without changing the model,
calibrator, thresholds, or external report. Selection is fixed before reading
examples and uses stable record IDs with deterministic score ordering.

The 21-record sample contains:

- the three highest calibrated human `likely_machine` scores in each of the
  three external domains (9 records);
- the three lowest calibrated machine `likely_human` scores for each of the
  two external generators (6 records); and
- one uncertain human nearest the machine boundary and one uncertain machine
  nearest the human boundary in each domain (6 records).

Ties use the lowest record ID. This is a purposive high-cost sample, not a
random sample or an estimator of cue prevalence. It is intended to identify
plausible visible failure modes that complement, but never replace, the
population metrics.

Prepare the ignored local excerpt packet and annotation template with:

```powershell
python scripts/review_external_errors.py --prepare
```

The packet contains each selected record's opening 240 and closing 120
collapsed-whitespace characters plus deterministic structural counts. It stays
under ignored `artifacts/error_review/`; the annotation template stays under
ignored `data/interim/`. Review notes must be generic observations of at most
160 characters and cannot contain quotations. Each record receives one or more
predeclared surface-cue codes, including an explicit `no_clear_surface_cue`
option.

After annotating every selected ID, finalize and independently reproduce the
text-free report with:

```powershell
python scripts/review_external_errors.py --finalize
python scripts/review_external_errors.py --verify-only
```

The committed report contains selected IDs, outcome metadata, cue codes,
generic reviewer notes, and aggregate counts. It contains no source excerpt or
full text. The annotations describe what is visible in a short review packet;
they do not establish why the model produced a score, make causal claims, or
generalize to all errors.

## Completed review

The real 21-record review and verification-only reproduction completed on
2026-08-10. It contains 9 human false-machine cases, 6 machine false-human
cases, and 6 uncertain-boundary cases. The sample spans 7 creative-writing, 5
news, and 9 student-essay records; 12 are human, 6 Claude, and 3 GPT-3.5 Turbo.

| Visible cue | Coded records |
| --- | ---: |
| Source-like specific detail | 15 |
| Formal or formulaic register | 11 |
| Academic-essay conventions | 9 |
| Narrative or dialogue style | 7 |
| Institutional or newswire style | 5 |
| First-person or personal voice | 4 |
| Citation or reference markers | 4 |
| Enumerative or heading structure | 2 |
| Short or fragmentary | 1 |

The human false-machine subset does not have one visible genre: its nine cases
split evenly among three creative narratives, three Reuters-style reports, and
three formal academic essays. Six contain source-like specific detail. This
small review therefore gives no basis for dismissing the false-machine cases
as one narrow formatting anomaly.

All six machine false-human cases show formal or formulaic prose. Four are
academic-style essays with subject-specific detail; two are first-person
speculative narratives. The six uncertain cases also span narrative, newswire,
and academic styles, and all six contain source-like specific details. These
patterns are compatible with source and register sensitivity, but the review
cannot prove which features caused a score.

The authoritative
[`ghostbuster_error_review_report.json`](../../data/metadata/ghostbuster_error_review_report.json)
contains stable IDs, outcome metadata, cue codes, generic notes, input hashes,
and the ignored packet identity. It contains no excerpts or source text.

## Review limitations

This is a deliberately score-extreme, stratified sample, so its cue counts are
not prevalence estimates. One reviewer coded bounded opening and closing
excerpts rather than full documents; there is no second-reviewer agreement
measure. The taxonomy captures visible surface patterns, not latent causes,
model-feature attribution, semantic quality, truth, or authorship. No
annotation changed any metric or policy decision.
