# Solution for Inundata: Mapping Floods in South Africa

1. rule based baseline LB 0.004810
2. simple xgboost baseline LB 0.004218
3. add precipitation rolling mean feature LB 0.00318
4. rolling mean center=True LB 0.00296
5. add diff and rolling mean of diff LB 0.00271
6. use gaussian smooth label regression as base margin LB 0.00252
7. train image model to classify flood vs non-flood locations and normalize probability LB 0.00245
