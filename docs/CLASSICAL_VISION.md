# Classical optical spot localization

This layer estimates a beacon centre from camera pixels only. Simulator target coordinates, object IDs, distractor identities, world position and optical-axis truth are not arguments of `VisionTracker.measure(frame, timestamp)`.

## Coordinates and common output

Image origin is top-left. `x`/`u` increases right, `y`/`v` increases down, and units are pixels. Camera centre is `(width/2, height/2)`. All algorithms retain floating-point subpixel coordinates.

`Measurement` is the shared `SpotMeasurement` contract. It contains position, validity, normalized classical quality, spot radius/spread, peak and integrated intensity, background statistics, `image_snr_estimate`, fit residual, algorithm/version identity, timestamp, optional 2×2 pixel covariance, raw quality features and an explicit failure reason. Missing values are `None`; they are not fabricated.

## Preprocessing and candidate association

Grayscale conversion is explicit. Optional Gaussian denoising and morphological opening are configured and stored. Background is estimated with the image median and `1.4826 × MAD`; a lower-image standard deviation fallback is used when MAD is zero.

Threshold strategies are:

- fixed: `T = configured DN`;
- relative peak: `T = B + f(max(I)-B)`;
- background sigma: `T = B + max(1 DN, k sigma_B)`.

Connected components are generated once as `SpotCandidate` objects. Each contains bounding box, area, peak, integrated intensity and approximate centroid—never simulator identity. The default association cost is

`0.55 normalized_distance + 0.20 normalized_size_change - 0.25 normalized_integrated_intensity`.

The distance reference is the prior legitimate measurement when available, otherwise image centre. `brightest` and `nearest_previous` policies are also explicit options.

## Localizers

### Legacy centroid

The original OpenCV contour implementation is preserved unchanged under `centroid` and `legacy_centroid`. It thresholds the image, chooses the largest contour between its original area bounds, then uses OpenCV spatial moments. It is the `DEFAULT_STABLE` vision method until benchmark evidence says otherwise.

### Binary centroid

For the selected component pixels:

`x_c = (1/N) sum(x_i)`, `y_c = (1/N) sum(y_i)`.

It is fast and interpretable, but threshold choice changes the selected shape.

### Intensity-weighted centroid

Weights are non-negative background-subtracted intensities `w_i=max(I_i-B,0)` inside the selected component:

`x_c=sum(x_i w_i)/sum(w_i)`, `y_c=sum(y_i w_i)/sum(w_i)`.

Near-zero total weight returns `ZERO_WEIGHT`, never arbitrary coordinates.

### Gradient-weighted centroid

Sobel derivatives produce `G=sqrt(G_x²+G_y²)`. With relative cutoff `G_min=f_g max(G)`, the exact implemented weight is:

`w(x,y) = (max(I-B,0)+1) G(x,y) 1[G>=G_min] 1[selected component]`.

The weighted-coordinate formula is then applied. This is the project’s documented implementation, not a claim of reproducing a named paper.

### Unrotated 2-D Gaussian fit

The model is

`I(x,y)=B+A exp(-0.5[((x-x0)/sigma_x)²+((y-y0)/sigma_y)²])`.

An intensity-moment initialization feeds bounded, damped Gauss–Newton. Background/amplitude, centre and log-spreads are fitted. Bounds prevent negative amplitude and unreasonable spread; damping increases after a rejected step. Invalid, singular or non-finite fits return `FIT_FAILED`. `fit_error` is ROI residual RMSE. When the normal matrix is usable, the `(x0,y0)` block of the residual-scaled parameter covariance is returned; otherwise covariance is `None`.

## Quality and confidence

`image_snr_estimate=(peak-background_median)/max(background_std,1 DN)` is an image-domain proxy, not electrical or optical receiver SNR. Stored raw features include area, equivalent radius, sigma, ellipticity, gradient energy, background statistics, saturation fraction, border clipping and candidate count/ambiguity.

The summary confidence is a transparent classical quality score: `snr/(snr+5) × shape_score × fit_score × sqrt(1/candidate_count) × clipping_penalty`, clamped to `[0,1]`. Raw components remain available so later estimators need not trust the scalar alone.

## Failures and complexity

Failures are `NO_CANDIDATE`, `LOW_SIGNAL`, `AMBIGUOUS_CANDIDATES`, `ZERO_WEIGHT`, `FIT_FAILED`, `OUT_OF_FRAME`, `SATURATED`, `INVALID_NUMERICS` or `UNKNOWN`. Empty/zero/non-finite images cannot produce NaN measurements. Centroids and Sobel processing are linear in ROI pixels; iterative Gaussian fitting costs more and reports iterations and latency.

No CNN, YOLO or learned correction is present. Dataset export is explicit and user-triggered; it groups samples by trajectory, scenario and seed for future leakage-safe splitting.
