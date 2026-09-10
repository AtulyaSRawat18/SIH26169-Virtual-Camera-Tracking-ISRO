# SIH Judge Demo Guide

## Start

From the repository root, run:

```powershell
.\launch_simulation.cmd
```

The launcher creates/uses `.venv`, installs missing Python packages, installs missing frontend packages, builds the dashboard, starts FastAPI, and opens `http://127.0.0.1:8000`. Press `Ctrl+C` in the launcher window to stop it. No external service, GPU, or satellite hardware is required.

## Recommended 3–5 minute flow

The six `DEMO_*` presets and the nominal startup case begin with coarse alignment already established, so the first visible PAT state is `LOCKED`. Acquisition-lab and other research presets still begin in `SEARCH`. Each live case shows a short note with the validated architecture and why it is retained for that condition.

1. Open **Live Demo** with `DEMO_1_NOMINAL_SAT_SAT`. Point out the analytical/numerical motion, LOS/FOV, actual sensor feed, PAT state, and pointing error.
2. Toggle **RAW** and **ANNOTATED**. Explain that tracking consumes the raw frame; overlays are drawn afterward. Enable **EVAL TRUTH** and point to its explicit evaluation-only label.
3. Open **Expert controls** and raise vibration or image noise. The simulation restarts with the same deterministic master seed and logs the changed value. Watch raw/estimator/pointing errors fluctuate rather than animate downward.
4. Select `DEMO_3_WEAK_BEACON_GROUND_SAT`. Show reduced image SNR, uncertainty, lock response, and optical consequence. CNN is an optional experiment, not the default.
5. Select `DEMO_4_DROPOUT_REACQUISITION`. At 4–6 seconds the beacon disappears; show invalid measurement, LOST/REACQUIRE events, covariance/prediction-based search, and recovery.
6. Open **Evidence**. Run the same-seed Before/After comparison or a three-seed sweep. Show improvement/degradation labels, confidence interval, failures, compatible history, and batch/run provenance.
7. Open the print-ready report. Finish with the checked-in 100-pair acceptance snapshot and the honest decision not to promote the more complex candidate.

Do not demonstrate every localizer, filter, controller, or search strategy. The Advanced view preserves those for questions.

## What each area proves

- **Live Demo:** the current physical state, camera observation, closed-loop response, PAT state, disturbance controls, and link consequence.
- **Experiment Lab:** exactly what physics/pipeline/seed will run; single, live, Monte Carlo, paired, distributions, and exported config.
- **Evidence:** current error flow, paired results, sensitivity, cumulative compatible data, worst failures, export, and report.
- **Advanced:** specialized Vision, Estimator, CNN, Predictor, Controller, and Acquisition labs.
- **About:** the technology stack, USP, AI role, and limitations.

## Saved configurations

- `DEMO_1_NOMINAL_SAT_SAT`
- `DEMO_2_HIGH_VIBRATION`
- `DEMO_3_WEAK_BEACON_GROUND_SAT`
- `DEMO_4_DROPOUT_REACQUISITION`
- `DEMO_5_AGGRESSIVE_UAV`
- `DEMO_6_COMBINED_STRESS`

They are numerical configurations, not scripted result animations. All use the stable classical pipeline by default, so missing optional AI weights cannot break the main demo.

## AI fallback

If an experimental CNN or temporal checkpoint is unavailable or fails a latency/OOD gate, switch to `DEFAULT_STABLE`. The intended engineering story is that AI augments classical physics, not that it replaces a working fallback. The final promotion gate retained centroid + KF + PID because the candidate did not justify its compute cost.

## Common judge questions

- **Is this flight ready?** No. It is a reproducible software PAT testbed; see the limitations in `FINAL_SYSTEM_OVERVIEW.md`.
- **Where is AI?** The CNN corrects present-frame spot localization and estimates uncertainty; GRU/LSTM predict short-horizon motion. Both are experimental because aggregate promotion gates failed.
- **Why OpenCV?** A beacon is a compact optical spot, so classical intensity/gradient/Gaussian localization is fast, interpretable, and an essential satellite-relevant baseline.
- **How is fairness maintained?** Paired pipelines reuse identical physical configuration and isolated subsystem seeds; ground truth scores outputs but never steers them.
- **What proved the result?** Open `evidence/final_acceptance_snapshot.json` and batch `a71b8273-19d6-49b0-9080-91fb5a9d765c`.
- **Why is link availability zero in stress?** The current loop is coarse pointing and does not include a fine-steering stage; this is a discovered limitation, not hidden data.
