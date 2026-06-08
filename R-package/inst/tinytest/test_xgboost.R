library(tinytest)
library(lignum)

if (requireNamespace("xgboost", quietly = TRUE)) {
  
  set.seed(42)
  X <- matrix(as.numeric(sample(0:100, 500, replace = TRUE)), nrow = 100, ncol = 5)
  y_clf <- sample(0:1, 100, replace = TRUE)
  y_reg <- runif(100)

  model_path <- tempfile(fileext = ".json")
  
  dtrain <- xgboost::xgb.DMatrix(data = X, label = y_clf)
  bst <- xgboost::xgb.train(
    params = list(objective = "binary:logistic", max_depth = 3),
    data = dtrain,
    nrounds = 10,
    verbose = 0
  )
  xgboost::xgb.save(bst, model_path)
  
  lg_model <- load_model(model_path, format = "xgboost")
  
  expected_raw <- predict(bst, dtrain, outputmargin = TRUE)
  actual_raw <- predict(lg_model, X, raw_score = TRUE, n_jobs = 1L)
  expect_equal(actual_raw, expected_raw, tolerance = 1e-4)
  
  expected_out <- predict(bst, dtrain, outputmargin = FALSE)
  actual_out <- predict(lg_model, X, raw_score = FALSE, n_jobs = 1L)
  expect_equal(actual_out, expected_out, tolerance = 1e-4)
  
  unlink(model_path)

  model_path <- tempfile(fileext = ".json")
  
  dtrain_reg <- xgboost::xgb.DMatrix(data = X, label = y_reg)
  bst_reg <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror", max_depth = 3),
    data = dtrain_reg,
    nrounds = 10,
    verbose = 0
  )
  
  xgboost::xgb.save(bst_reg, model_path)
  
  lg_model_reg <- load_model(model_path, format = "xgboost")
  
  expected_reg <- predict(bst_reg, dtrain_reg)
  actual_reg_raw <- predict(lg_model_reg, X, raw_score = TRUE, n_jobs = 1L)
  actual_reg_out <- predict(lg_model_reg, X, raw_score = FALSE, n_jobs = 1L)
  
  expect_equal(actual_reg_raw, expected_reg, tolerance = 1e-4)
  expect_equal(actual_reg_out, expected_reg, tolerance = 1e-4)
  
  unlink(model_path)
  
  model_path <- tempfile(fileext = ".json")
  y_multi <- sample(0:2, 100, replace = TRUE)
  dtrain_multi <- xgboost::xgb.DMatrix(data = X, label = y_multi)
  bst_multi <- xgboost::xgb.train(
    params = list(objective = "multi:softprob", num_class = 3, max_depth = 2),
    data = dtrain_multi,
    nrounds = 2,
    verbose = 0
  )
  
  xgboost::xgb.save(bst_multi, model_path)
  
  expect_error(load_model(model_path, format = "xgboost"))
  unlink(model_path)
}