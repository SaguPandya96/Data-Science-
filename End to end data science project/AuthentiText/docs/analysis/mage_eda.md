# MAGE development exploratory analysis

This analysis profiles only the sanitized train and validation partitions.
Published test text and labels are deliberately excluded so exploratory
findings cannot influence feature choices or thresholds. The complete,
text-free aggregate report is
[`mage_eda_report.json`](../../data/metadata/mage_eda_report.json).

## Reproduce

```powershell
python scripts/analyze_eda.py
python scripts/analyze_eda.py --check
```

The command verifies all sanitized split files by their recorded identities,
but computes text statistics only for 287,843 train and 50,509 validation
records. The 50,567 published test records are marked excluded from EDA.

## Composition

The analysis population contains 338,352 records: 112,562 human target 0 and
225,790 machine target 1. It spans nine human source values and 279 machine
source values. Human sources contain 5,351 to 34,484 development records each;
machine sources contain 346 to 900. This asymmetry means random record-level
metrics can be dominated by many generator-specific source values even though
the human class has fewer sources.

| Domain | Human | Machine | Total |
| --- | ---: | ---: | ---: |
| CMV | 6,640 | 22,870 | 29,510 |
| ELI5 | 19,851 | 28,695 | 48,546 |
| HellaSwag | 6,411 | 27,508 | 33,919 |
| ROCStories | 6,550 | 28,687 | 35,237 |
| Scientific generation | 6,966 | 20,997 | 27,963 |
| SQuAD | 18,327 | 22,402 | 40,729 |
| TL;DR | 5,351 | 22,237 | 27,588 |
| XSum | 7,982 | 29,304 | 37,286 |
| Yelp | 34,484 | 23,090 | 57,574 |

Among machine records, 196,432 use the upstream `continuation` strategy,
14,401 use `specified`, and 14,957 use `topical`. These counts describe dataset
composition; they are not performance measurements.

## Length

Whitespace tokens use Python `str.split()` and are not a model-tokenizer count.

| Target | Records | Median tokens | Mean tokens | P95 tokens | Median characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human (0) | 112,562 | 110 | 169.780 | 518 | 648 |
| Machine (1) | 225,790 | 108 | 207.675 | 784 | 633 |

The medians are similar, but the machine distribution has a substantially
longer upper tail. Records above 512 whitespace tokens account for 12.7034% of
machine examples and 5.1012% of human examples, an absolute gap of 7.6022
percentage points. A later length-only diagnostic baseline will measure how
predictive this artifact is. It is not a suitable production detector.

Short records below 50 whitespace tokens are also somewhat more common among
machine examples (23.6848%) than human examples (21.5739%), a 2.1109-point
gap. Overall development lengths range from 6 to 10,090 whitespace tokens,
with median 109 and P95 747.

## Predefined structural indicators

Before scanning, the analysis defines flags for URLs, Markdown fences, heading
markers, line breaks, non-ASCII text, repeated spaces, surrounding whitespace,
and the two length thresholds. All seven non-length flags occur zero times in
the sanitized development data. That is a measured property of this release,
not a general statement about human or machine text.

## Implications and limits

- Report class-aware and domain/source-stratified results; aggregate accuracy
  cannot establish generalization.
- Include a majority baseline and a length-only diagnostic before lexical
  TF-IDF logistic regression.
- Do not pass source, domain, generator, partition, record ID, or target
  metadata into text models.
- Keep the published test uninspected until model configuration and threshold
  policy are fixed from train and validation.

The report is descriptive and cannot establish causality. It contains no raw
text or record identifiers.
