# Monte Carlo and paired comparison method

The batch runner calls the same SimulationEngine used live. It omits Three.js,
DOM work, annotations, JPEG encoding and WebSocket publication. Image formation
and OpenCV remain because they are the system under test.

For each run, SHA-256 derives subsystem seeds from master seed, run index and a
stable subsystem name. Camera-noise draws cannot change vibration, dropout,
manoeuvre or distractor timing. The database records the master seed, effective
run seed, run index through the resolved config, and every subsystem seed.

Parameter distributions accept `uniform`, `normal` or `choice` and a dotted
configuration path. Every sampled value is retained with its individual run.
Summary aggregation reports count, mean, median, population standard deviation,
minimum, maximum, fifth and 95th percentile for continuous metrics, plus
acquisition/lock/reacquisition success and false-lock probability.

Paired comparison first verifies identical resolved physical configurations.
Variant A and B then use the same master seed and run index, producing common
environmental random numbers. Only registered algorithm settings may differ.
Controllers can change later camera geometry, which is the intended closed-loop
effect; their underlying disturbance samples remain paired.

Default base seed is 42 and default requested batch size is 100. UI offers 10,
100 and a small validation option. Large batches store summaries by default;
`summary_only=false` stores bounded per-frame telemetry. Long batches currently
run sequentially in the API process and should move to a background worker next.
