library(tinytest)
library(lignum)

if (requireNamespace("lightgbm", quietly = TRUE)) {
    
  set.seed(42)
  X <- matrix(as.numeric(sample(0:100, 500, replace = TRUE)), nrow = 100, ncol = 5)
  y <- sample(0:1, 100, replace = TRUE)
  
  dtrain <- lightgbm::lgb.Dataset(data = X, label = y)
  params <- list(objective = "binary", max_depth = 3, min_data_in_leaf = 1)
  lgb_model <- lightgbm::lgb.train(params = params, data = dtrain, nrounds = 10, verbose = -1)
  
  json_path <- tempfile(fileext = ".json")
  bin_path <- tempfile(fileext = ".bin")
  
  writeLines(lgb_model$dump_model(), json_path)
  
  lg_model_json <- load_model(json_path, format = "lightgbm")
  
  save_model(lg_model_json, bin_path)
  lg_model_bin <- load_model(bin_path, format = "binary")
  
  expected_raw <- predict(lgb_model, X, type = "raw")
  actual_json_raw <- predict(lg_model_json, X, raw_score = TRUE, n_jobs = 1L)
  actual_bin_raw <- predict(lg_model_bin, X, raw_score = TRUE, n_jobs = 1L)
  
  expected_prob <- predict(lgb_model, X, type = "response")
  actual_json_prob <- predict(lg_model_json, X, raw_score = FALSE, n_jobs = 1L)
  actual_bin_prob <- predict(lg_model_bin, X, raw_score = FALSE, n_jobs = 1L)

  expect_equal(actual_json_raw, expected_raw, tolerance = 1e-5)
  expect_equal(actual_json_prob, expected_prob, tolerance = 1e-5)
  
  expect_equal(actual_bin_raw, actual_json_raw, tolerance = 1e-7)
  expect_equal(actual_bin_prob, actual_json_prob, tolerance = 1e-7)
  
  unlink(json_path)
  unlink(bin_path)
}