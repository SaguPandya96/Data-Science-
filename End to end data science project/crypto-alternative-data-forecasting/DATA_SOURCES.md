# Data sources and point-in-time notes

| Source | Variables | Coverage used | Frequency | Access / limits | Licensing | Quality / point-in-time caveat |
|---|---|---|---|---|---|---|
| Yahoo Finance chart endpoint | BTC-USD open, high, low, close, volume | 2022-01-01–2026-06-30 | Daily | Public endpoint; undocumented limits and throttling may change | Yahoo terms; personal/research use should be confirmed | Vendor corrections are possible; no market cap; daily timestamps normalized to UTC |
| GDELT DOC 2.0 | Headline, publication/seen timestamp, URL, domain, language, source country | Rolling recent window available at run time | Article | No key; 250 records/query; chunked and retried | GDELT/open-source terms; publisher article rights remain with publishers | `seendate` is a GDELT observation time, not guaranteed first publication; historical DOC search is limited |
| VADER | Positive/neutral/negative proportions and compound score | Every cached headline | Headline | Local, no rate limit | MIT | General-language lexicon misses finance nuance, sarcasm, and entity context |
| FinBERT (optional) | Class probabilities | Not used in reference run | Headline | Hugging Face download; large model | Model-card license/terms | More compute; domain mismatch remains possible; outputs must be cached |

The repository remains runnable after an outage when nonempty raw caches exist. The checked-in reference data is a reproducibility snapshot, not a claim that either upstream source is immutable.
