# Decision log

## Why I did not use a ready-made fraud CSV

The small labeled datasets I found made the modeling step easy but left very little to investigate. Criteo's data gave me real campaign, click, conversion, timing, and cost structure. The tradeoff is that it has no fraud truth. I preferred that honest limitation over pretending a convenient label represented the real problem.

## Why I sampled the beginning of the file

The compressed source is more than 600 MB and expands to roughly 2.4 GB. Streaming the first 500,000 rows gives a reproducible local run without copying the whole archive. Because the source is time-sorted, this is an early-period slice, not a representative random sample. I call that out everywhere results appear.

## Why I added simulations

Without labels, I needed a falsifiable test. I added three patterns to untouched observed rows and recorded every episode in a manifest. I also added a clean popularity spike because a detector that flags every surge is not useful.

## Why I used 30-minute campaign windows

Row-level predictions would mostly learn whether an individual impression was clicked. The behavior I cared about was collective: volume, concentration, timing, and downstream quality. Thirty minutes was short enough to preserve bursts and long enough to avoid mostly-empty campaign groups in the local sample.

## Why I kept the model small

I wanted to understand every transformation. A class-weighted logistic model gave me a useful supervised baseline, and a robust empirical distance gave me an unsupervised comparison. I combined them experimentally, but the held-out result showed that the hybrid was not automatically better. That is a result, not something to conceal.

## What I would change next

1. Run the full 30-day source and hold out complete campaigns.
2. Build campaign-specific seasonal baselines.
3. Add a second negative control that mimics a budget launch.
4. Tune the anomaly component for the missed adaptive pattern.
5. Ask another reviewer to inspect cases without access to simulation labels.
