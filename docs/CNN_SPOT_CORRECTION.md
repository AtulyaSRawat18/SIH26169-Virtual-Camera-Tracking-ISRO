# CNN spot correction

The learned component is a residual corrector, not a detector and not a replacement
for OpenCV. The deployed path is:

`raw frame → classical localizer → fixed 32×32 ROI → tiny CNN → (Δu, Δv, log σ²u, log σ²v)`

The corrected measurement is `(u+Δu, v+Δv)`. Its calibrated diagonal covariance is
passed to AKF-R. Ground truth is never an input to the runtime corrector; it is used
only when constructing labels and computing benchmark metrics.

The model has 23,980 parameters and is trained with heteroscedastic Gaussian negative
log likelihood. The input contract uses the exact same fixed ROI extractor in dataset
generation and deployment. Auxiliary inputs describe the classical fit, SNR,
ellipticity, saturation, ambiguity, and border clipping.

Safety gates reject missing checkpoints, malformed output, excessive correction,
excessive uncertainty, invalid ROI, excessive latency, and auxiliary-feature OOD.
Every rejection exposes a reason and returns the unchanged classical measurement.
The dashboard renders the classical and corrected points separately.

Current status: implemented and runnable as `AI_MEASUREMENT_HYBRID`, but not promoted.
The v4 held-out RMSE was 0.08690 px versus 0.08675 px for OpenCV, and OOD performance
was materially worse. This is a valid negative result: classical subpixel localization
is already near the renderer's accuracy limit.
