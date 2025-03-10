# Solution for Inundata: Mapping Floods in South Africa

This project aims to predict flood occurrences in South Africa.  The approach involves iterative improvements using feature engineering and modeling techniques.

| Step | Description                                                                                                         | LB Score  |
|------|---------------------------------------------------------------------------------------------------------------------|-----------|
| 1    | Rule-based baseline (based on flood days and locations only)                                                            | 0.00481  |
| 2    | Simple XGBoost baseline (using initial features without feature engineering)                                           | 0.00439  |
| 3    | Add precipitation rolling mean feature (various window sizes from 2 to 240)                                           | 0.00318   |
| 4    | Rolling mean with `center=True` (considers both past and future values)                                              | 0.00296   |
| 5    | Add difference and rolling mean of difference (lags of 2, 8, 14, 28 days; various window sizes)                      | 0.00271   |
| 6    | Use Gaussian smooth label regression as base margin                                                  | 0.00252   |
| 7    | Train image model (YOLO) to classify flood vs non-flood locations and normalize probability                         | 0.00245   |

**Reproduce the best solution:**

```python

pip install -r requirements.txt

# edit PATH in const.py

# train and predict image models
python create_cv.py
./train_cls.sh
./infer_cls.sh

# train gaussian smooth label regression model
python rv.py

# train final classification model
python cv.py
```

**Progression and Results:**

1.  **Rule-based baseline (LB: 0.004810):**  An initial baseline based on flood days and locations only.

2.  **Simple XGBoost baseline (LB: 0.00439):**  A basic XGBoost model is trained, using the initial features in the dataset without feature engineering.

3.  **Add precipitation rolling mean feature (LB: 0.00318):** This significant improvement comes from adding rolling mean features of precipitation.  The `fe` function (`cv.py`, lines 21-45) calculates rolling means for various window sizes (`w`):
    *   `for w in list(range(2,100,2)) + list(range(100, 250, 10)):`  This creates rolling mean features with windows from 2 to 98 (step 2) and 100 to 240 (step 10).  These are stored as columns named `rm_{w}_precipitation`.
    *   This captures trends in precipitation over different time scales.

4.  **Rolling mean center=True (LB: 0.00296):**  The `center=True` argument in the `rolling()` function within `fe` (`cv.py`, line 23) is crucial.  Centering the rolling window means that the average at a given time point considers both past *and* future values.  This reduces lag and provides a smoother representation of the precipitation trend.

5.  **Add diff and rolling mean of diff (LB: 0.00271):**  This step adds features related to the *change* in precipitation.  The `fe` function (`cv.py`, lines 27-41) calculates lagged differences and their rolling means:
    *   `df[f'rainfall_lag_{lag}_{col}' = ... .diff(lag).fillna(0)`:  Calculates the difference in precipitation between the current day and `lag` days prior.  `fillna(0)` handles the initial `NaN` values.
    *   Nested loops create `lag_rm_{w}_{lag}_precipitation` features.  These are rolling means (windows `w`) of the lagged differences (`lag`).  This captures the trend of precipitation *changes* over different time scales.  This is done for lags of 2, 8, 14, and 28 days, and a variety of window sizes.

6.  **Use Gaussian smooth label regression as base margin (LB: 0.00252):** This step employs a Gaussian smoothing technique to transform the binary flood labels (0 or 1) into a continuous, "soft" target variable.  This smoothed representation is then used as the target variable for an XGBoost regression model. This approach is beneficial because it provides a more nuanced representation of flood risk and helps the model learn a smoother decision boundary. The predictions of this regression model is then used as the base margin for the final classification model


7.  **Train image model to classify flood vs non-flood locations and normalize probability (LB: 0.00245):**  
A YOLO image classification model (`train_cls.py`) is trained to distinguish flood-prone locations.  Training uses 128x128 images, data augmentation, and 10 epochs.  The model's output probabilities `flood_a1` are for each location, which is used to select flood locations and normalize the probabilities.


**Overall Approach:**

The solution uses a multi-stage approach:

1.  **Feature Engineering:**  Extensive feature engineering is performed, focusing on:
    *   Rolling means of precipitation and lagged precipitation differences.

2.  **Modeling:**
    *   An initial rule-based baseline.
    *   XGBoost models are trained with increasingly complex features.
    *   Use Gaussian smooth label regression as base margin to improve model performance.
    *   An image model is used for location-based classification.

3.  **Normalization:**  The `normalize` function adjusts predictions, using the image model output to scale probabilities within each location.
