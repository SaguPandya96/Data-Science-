# Versioned inference contract

`AuthentiTextPredictor` is the single text-to-decision boundary for the frozen
word TF-IDF model, isotonic calibrator, and abstention thresholds. It verifies
versioned report identities before loading artifacts and returns a JSON-safe
dictionary with schema version 1.

## Local use

```powershell
python scripts/predict_text.py --input path/to/utf8-text.txt
Get-Content path/to/utf8-text.txt -Raw | python scripts/predict_text.py
```

The CLI emits the prediction to standard output and structured errors to
standard error. It does not write the submitted text. Shell history and source
files are outside the application's privacy boundary, so file/stdin use is
preferred over command-line text arguments; no text argument is offered.

## Input rules

- Input must be a string with at least one non-whitespace character.
- NUL characters are rejected.
- The maximum is 100,000 Unicode code points.
- Text is preserved for the vectorizer; it is not truncated or rewritten.
- Whitespace-token counts use Python `str.split()` only for warnings and
  metadata, not as word-model input.

Errors have stable codes: `text_type`, `text_blank`, `text_null`, and
`text_too_long`. Artifact/report failures use separate codes and prevent
readiness rather than serving an unverified model.

## Response

The response includes:

- `category`: `likely_human`, `uncertain`, or `likely_machine`;
- `calibrated_machine_likelihood`: the isotonic score rounded to six decimals;
- `raw_model_score`: the lexical logistic score rounded to six decimals;
- `evidence_quality`: `standard` or `low`;
- character and whitespace-token counts;
- the exact frozen thresholds;
- structured warnings and limitations;
- dataset revision plus base-model and calibration SHA-256 identities.

The submitted text, record hashes, inferred source/domain, and feature weights
are not returned.

The calibrated number estimates target frequency under the MAGE validation
setup. It is not a universal probability that a person or model authored the
input. The category remains an evidence label, never proof.

## Warnings

Inputs below 50 whitespace tokens receive `short_text_low_evidence`; frozen
test decisive accuracy in that band was only 0.588441. Inputs outside the 6 to
10,090 token development range receive `length_outside_development_range`.

The development EDA observed zero instances of seven predefined formatting
flags. Inputs containing any receive one `format_out_of_profile` warning with
the detected flag names:

- Markdown fence;
- line break;
- non-ASCII character;
- repeated ASCII space;
- HTTP/HTTPS marker;
- surrounding whitespace;
- leading Markdown heading marker.

This warning reflects the measured data profile, not a judgment that the
formatting is suspicious. A line break from piping a file can therefore trigger
it.

Warnings do not change the frozen category. Introducing a post-test category
override would alter the evaluated policy; any such change must be a new model
version with its own validation and test cycle.

## Artifact integrity

Startup requires:

1. the training report's word-model artifact size and SHA-256 to match;
2. the calibration report's artifact size and SHA-256 to match;
3. model types to match the expected contract;
4. the calibration artifact's recorded base-model SHA-256 to match the loaded
   model;
5. `0 <= human_threshold < machine_threshold <= 1`.

Any failure stops loading. There is no silent fallback or uncalibrated response.

## Known limitations

Every response states that the estimate is not authorship proof, development is
English MAGE-specific, and errors vary by domain and length. The frozen test
results, including 5.2391% human false-machine, 5.9880% machine false-human,
and 57.0708% uncertain rates, remain the governing evidence for this version.
