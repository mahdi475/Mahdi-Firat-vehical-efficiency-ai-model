# Decisions

## Dataset

Use UCI Auto MPG as the primary dataset. It is small, documented, licensed under
CC BY 4.0, and maps directly to a regression task.

## Models

Compare `DummyRegressor`, `LinearRegression`, `DecisionTreeRegressor`, and
`RandomForestRegressor`. Use random forest as the default final model only if it is
compatible with the lab-model rule for the course presentation.

## Application Scope

Build one input/output demo page. The app supports the presentation; it is not the
main research contribution.

