# Price data

Put price files for replays here. They stay out of git because most data vendors don't allow
redistribution.

The format is a CSV with one row per print:

```text
ts,instrument,price
2026-11-13T15:30:00-05:00,VOO,561.20
2026-11-13T15:30:00-05:00,BTC-USD,98690.43
```

`ts` must carry a UTC offset. Instrument names must match the ones in the model files. Leave
out equity rows while their market is closed, so the replay sees the same gaps the live engine
would. One engine cycle runs per distinct timestamp, so 30-minute or 1-minute bars both work.
