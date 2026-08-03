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

## Why I added an observed-only expected-behavior track

The controlled experiment answered whether the pipeline could recover patterns I planted. It did not answer which original windows deserved investigation. For that question, I removed every simulation-touched window and modeled the outcomes that the source actually contains: clicks and conversion-linked impressions.

I call the output a traffic-quality review queue, not a fraud queue. The source has no invalid-traffic truth, and an unexpected residual has many possible explanations.

## Why the campaign baseline only looks backward

A baseline that includes the current or future window would leak the answer into the feature. I calculate rolling means from the previous 20 campaign windows in SQLite. New campaigns fall back to global values calculated only from the training period, and the evidence summary calls out limited history.

## Why I used two expected-outcome models

Click behavior and downstream conversion behavior answer different questions. The click model avoids current-window click features. The conversion model is evaluated later in the funnel, so it can use the clicks already observed in the window. Keeping the models separate makes both the timing and the residual interpretation clearer.

## Why the observed score stops at review

The combined score uses click surprise, conversion shortfall, campaign-relative volume, concentration, and transformed cost exposure. Those are useful prioritization signals, but none proves intent or policy violation. I therefore publish only `monitor` and `review_traffic_quality`; no automatic throttle or block is available in this track.

## What I would change next

1. Run the full 30-day source and compare time-based replay with holding out complete campaigns.
2. Add day-of-week and longer seasonal baselines once the full period is available.
3. Add a second negative control that mimics a budget launch.
4. Check review-queue stability under traffic and calibration drift.
5. Ask another reviewer to inspect both observed and experimental cases without hidden context.
