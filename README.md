# Solution for Inundata: Mapping Floods in South Africa

This project aims to predict flood occurrences in South Africa.  The approach involves iterative improvements using feature engineering and modeling techniques.

**Progression and Results:**

1.  **Rule-based baseline (LB: 0.004810):**  An initial baseline based on flood days and locations only.

2.  **Simple XGBoost baseline (LB: 0.004218):**  A basic XGBoost model is trained, using the initial features in the dataset without feature engineering.

3.  **Add precipitation rolling mean feature (LB: 0.00318):** This significant improvement comes from adding rolling mean features of precipitation.  The `fe` function (`cv.py`, lines 21-45) calculates rolling means for various window sizes (`w`):
    *   `for w in list(range(2,100,2)) + list(range(100, 250, 10)):`  This creates rolling mean features with windows from 2 to 98 (step 2) and 100 to 240 (step 10).  These are stored as columns named `rm_{w}_precipitation`.
    *   This captures trends in precipitation over different time scales.

4.  **Rolling mean center=True (LB: 0.00296):**  The `center=True` argument in the `rolling()` function within `fe` (`cv.py`, line 23) is crucial.  Centering the rolling window means that the average at a given time point considers both past *and* future values.  This reduces lag and provides a smoother representation of the precipitation trend.

5.  **Add diff and rolling mean of diff (LB: 0.00271):**  This step adds features related to the *change* in precipitation.  The `fe` function (`cv.py`, lines 27-41) calculates lagged differences and their rolling means:
    *   `df[f'rainfall_lag_{lag}_{col}' = ... .diff(lag).fillna(0)`:  Calculates the difference in precipitation between the current day and `lag` days prior.  `fillna(0)` handles the initial `NaN` values.
    *   Nested loops create `lag_rm_{w}_{lag}_precipitation` features.  These are rolling means (windows `w`) of the lagged differences (`lag`).  This captures the trend of precipitation *changes* over different time scales.  This is done for lags of 2, 8, 14, and 28 days, and a variety of window sizes.

6.  **Use Gaussian smooth label regression as base margin (LB: 0.00252):**  The provided code snippets *don't* explicitly show Gaussian smoothing.  However, based on the description, it indicates using the output from the earlier, non-normalized models to be used in a label regression step.

7.  **Train image model to classify flood vs non-flood locations and normalize probability (LB: 0.00245):**  This step utilizes an image model (details not in the provided code).  The `normalize` function (`cv.py`, lines 11-19) is key here:
    *   `df['ls'] = df.groupby('location')[col].transform('sum')`: Calculates the sum of the target variable (`col`, the flood probability) for each location.
    *   `df['lx'] = df.groupby('location')[col].transform('max')`: Calculates the maximum of the target variable for each location.
    *    A mask is applied using `flood_a1`.
    *   `df.loc[mask,col] = df.loc[mask,col]/df.loc[mask,'ls']`:  This normalizes the target variable by dividing by the sum of the target *for each location*, but only where `flood_a1` is greater than `confidence`.  This adjusts the probabilities based on the image model's classification, scaling them within each location.


**Overall Approach:**

The solution uses a multi-stage approach:

1.  **Feature Engineering:**  Extensive feature engineering is performed, focusing on:
    *   Rolling means of precipitation and lagged precipitation differences.
    *   Days since the last rain.
    *   Soil moisture estimation.
    *   Time to peak value.

2.  **Modeling:**
    *   An initial rule-based baseline.
    *   XGBoost models are trained with increasingly complex features.
    *   An image model (details not provided) is used for location-based classification.

3.  **Normalization:**  The `normalize` function adjusts predictions, using the image model output to scale probabilities within each location.
