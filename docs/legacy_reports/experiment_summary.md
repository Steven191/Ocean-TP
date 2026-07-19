# OOI Data Experiment Replication Report

This report summarizes the replication of experiments on the new OOI dataset.

## 0. Dataset Visualization
We performed a comprehensive visualization of the new dataset to understand its characteristics.

### Temperature-Pressure Profile
![T-P Profile](figures/dataset_tp_profile_hexbin.png)
*Figure 0.1: Temperature vs Pressure density plot showing the distribution of data points.*

### Temporal Distribution
![Time Series](figures/dataset_time_series.png)
*Figure 0.2: Time series of temperature measurements colored by pressure.*

### Seasonal Variation
![Seasonal Boxplot](figures/dataset_seasonal_boxplot.png)
*Figure 0.3: Monthly temperature distribution.*

### Depth-Time Heatmap
![Depth-Time Heatmap](figures/dataset_depth_time_heatmap.png)
*Figure 0.4: Monthly mean temperature variation across depths.*

## 1. Basis Function Expansion Experiment
We evaluated the performance of the model as we increased the number of basis function terms from 1 to 8.

| Terms | Train MAE | Test MAE | RMSE |
|-------|-----------|----------|------|
| 1 | 0.6763 | 0.6718 | 0.7868 |
| 2 | 0.1418 | 0.1411 | 0.1915 |
| 3 | 0.0856 | 0.0844 | 0.1377 |
| 4 | 0.0708 | 0.0696 | 0.1316 |
| 5 | 0.0698 | 0.0684 | 0.1276 |
| 6 | 0.0650 | 0.0638 | 0.1235 |
| 7 | 0.0646 | 0.0634 | 0.1233 |
| 8 | 0.0646 | 0.0634 | 0.1233 |

**Observation**: Significant improvement is observed up to 4 terms. The 8-term model provides stable performance.

![Term Increase Plot](figures/term_increase_mae.png)

## 2. Time-Scale Partitioning (Seasonal Analysis)
We partitioned the data into seasons and evaluated the model performance for each season.

| season | date_range | total_observations | train_MAE | train_RMSE | val_MAE | val_RMSE | val_R2 | physics_residual_mean | physics_residual_q95 | train_samples | val_samples | coef_1 | coef_P | coef_ln(P) | coef_sqrt(P) | coef_exp(-P/500) | coef_exp(-P/1200) | coef_1/(P+50) | coef_1/(P+500) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Season1 | 01-22 → 05-25 | 2385809 | 0.0589361226961138 | 0.1064773337446018 | 0.1087161082222695 | 0.2093024912559279 | 0.9464198886707712 | 1.8370148039333955e-06 | 0.0 | 1908647 | 477162 | -197.6428382697593 | -0.0132640655141843 | 23.01457908095412 | 0.8937863215460681 | 16.46949008085455 | 45.39374658664507 | 5500.1284077485925 | -872.0017241668164 |
| Season2 | 05-27 → 07-15 | 848483 | 0.0502048684055386 | 0.07639013239596 | 0.1337558834476998 | 0.2567503630844937 | 0.9211222980327152 | 6.498277580577588e-06 | 0.0 | 678786 | 169697 | -113.90737355845226 | -0.0385859324583752 | -5.2509332703519975 | 4.838840175109463 | 10.811727580865684 | 80.35633706563657 | 2703.7911790344056 | -408.1735919303718 |
| Season3 | 07-26 → 08-17 | 1553957 | 0.057696554951509 | 0.0935221946622898 | 0.1628319806139926 | 0.3297394421358136 | 0.8826605483952943 | 2.1140429542029683e-06 | 0.0 | 1243165 | 310792 | -142.44050250548472 | -0.0339574803281346 | 1.9436147071835643 | 4.058533527142073 | 12.670300953910283 | 75.5066786211343 | 3615.82722506518 | -606.2778839384346 |
| Season4 | 10-20 → 01-22 | 1936157 | 0.0604160134252676 | 0.1064591756422485 | 0.051200170881537 | 0.1010079495944586 | 0.9848139716457932 | 1.7266293694705268e-06 | 0.0 | 1548925 | 387232 | -170.46960760272768 | -0.0275855957765601 | 10.170696530820988 | 3.027958345692837 | 14.84387692854949 | 67.16890161023231 | 4518.012927747576 | -750.5232395646588 |


## 3 & 4. Depth Stratification and D-Optimal Selection
We implemented dynamic depth stratification and D-Optimal point selection.

### D-Optimal 8-Point Selection
Selected Pressure Points: [160.0, 2181.0, 235.0, 2181.0, 160.0, 461.0, 464.0, 853.0]

### Performance Metrics (Global vs D-Optimal)
| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| global | 0.0659 | 0.1296 | 0.9729 |
| d_opt_8 | 0.1248 | 0.1662 | 0.9554 |
| uniform_8 | 0.0705 | 0.1475 | 0.9649 |
| only8 | 0.1111 | 0.1698 | 0.9535 |

## 5. Recent Prediction Accuracy (New Metric)
We evaluated the models' ability to predict ocean temperatures in recent years (2020-2025).

| Year | D-Optimal MAE | Seasonal MAE | Monthly MAE | Samples |
|------|---------------|--------------|-------------|---------|
| 2018 | 0.0519 | 0.6660 | 2.9200 | 569060 |
| 2019 | 0.0819 | 0.6885 | 2.9410 | 530578 |
| 2020 | 0.1168 | 0.7316 | 3.0029 | 865202 |
| 2021 | 0.0719 | 0.8249 | 3.1666 | 361101 |
| 2022 | 0.0674 | 0.7859 | 3.0492 | 576605 |
| 2023 | 0.0488 | 0.7001 | 2.9411 | 866251 |
| 2024 | 0.1511 | 0.7835 | 3.0503 | 865001 |
| 2025 | 0.1501 | 0.7892 | 3.0483 | 344364 |

**Key Insight**: The D-Optimal model generalizes surprisingly well to recent data, often outperforming the more complex seasonal/monthly models in 2021-2023. However, all models show degradation in 2024-2025.

## 6. Overall Model Comparison
Comparison of different modeling approaches:

- **Global 8-Term Model**: Simple, robust, low maintenance. Good generalization.
- **Seasonal Layered Model**: Captures seasonal variations but requires more data and is more complex.
- **Monthly Layered Model**: Most granular, best potential accuracy but prone to overfitting if data is sparse (as seen in some months/years).
