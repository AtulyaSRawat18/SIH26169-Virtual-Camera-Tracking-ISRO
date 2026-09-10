# Temporal prediction

Prediction is a distinct post-estimation stage:

`measurement → AKF-R state/covariance → predictor → predicted future pixel → controller`

The predictor receives only current and historical estimator states, measurement
quality, timestamps, and a physical horizon. It never receives future truth. Available
algorithms are NONE, constant velocity (CV), robust constant acceleration (CA), GRU,
LSTM, and a GRU residual on CV.

Neural outputs include a future pixel and diagonal uncertainty. Missing weights,
insufficient history, invalid output, excessive displacement, or excessive uncertainty
produce an observable CV fallback. The current checkpoints use 12 frames and support
20, 50, 100, and 200 ms horizons.

FF-PID adds a bounded feed-forward angular-rate term derived from the predicted image
velocity and focal length. Learned uncertainty scales this term between zero and one.
The physical gimbal remains the final authority and still enforces rate, acceleration,
position, lag, and bias limits.

Current status: all variants are runnable, but learned prediction is not globally
promoted. Use `PREDICTIVE_GRU_EXPERIMENT` for a labelled demonstration; use
`DEFAULT_STABLE` for the evidence-backed default.
