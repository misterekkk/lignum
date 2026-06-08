library(tinytest)
library(lignum)

if (requireNamespace("lightgbm", quietly = TRUE)) {
  
  set.seed(42)
  X <- matrix(as.numeric(sample(0:100, 500, replace = TRUE)), nrow = 100, ncol = 5)
  y <- sample(0:1, 100, replace = TRUE)
  
  model_path = tempfile(fileext = ".json")
  
  dtrain <- lightgbm::lgb.Dataset(data = X, label = y)
  params <- list(objective = "binary", max_depth = 3, min_data_in_leaf = 1)
  bst <- lightgbm::lgb.train(params = params, data = dtrain, nrounds = 15, verbose = -1)
  
  writeLines(bst$dump_model(), model_path)
  
  lg_model <- load_model(model_path, format = "lightgbm")
  
  expected_raw <- predict(bst, X, type = "raw")
  actual_raw <- predict(lg_model, X, raw_score = TRUE, n_jobs = 1L)
  expect_equal(actual_raw, expected_raw, tolerance = 1e-5)
  
  expected_prob <- predict(bst, X, type = "response")
  actual_prob <- predict(lg_model, X, raw_score = FALSE, n_jobs = 1L)
  expect_equal(actual_prob, expected_prob, tolerance = 1e-5)
  
  unlink(model_path)
  
  model_path = tempfile(fileext = ".json")
  y_multi <- sample(0:2, 100, replace = TRUE)
  dtrain_multi <- lightgbm::lgb.Dataset(data = X, label = y_multi)
  bst_multi <- lightgbm::lgb.train(
    params = list(objective = "multiclass", num_class = 3, max_depth = 2),
    data = dtrain_multi, nrounds = 2, verbose = -1
  )
  
  writeLines(bst_multi$dump_model(), model_path)
  
  expect_error(load_model(model_path, format = "lightgbm"))
  unlink(model_path)
}