#' @details 
#' Lignum deliveres fast interference for decision-tree based models.
#' @docType package
#' @name lignum
#' @useDynLib lignum, .registration = TRUE
NULL

#' Load a trained model from a JSON or Lignum binary file.
#'
#' @title Load a Lignum Model
#' @param filepath A string specifying the path to the model file.
#' @param format A string specifying the format of the model file. 
#'   Options are \code{'auto'}, \code{'lignum'}, \code{'binary'}, \code{'lightgbm'}, \code{'xgboost'}. 
#'   If \code{'auto'} (the default), the function attempts to load binary file.
#'
#' @return An object of class \code{lignum_model} ready for inference.
#' @export
#'
#' @examplesIf FALSE
#' model <- load_model("model_xgboost.json", format = "xgboost")
load_model <- function(filepath, format = "auto") {
  filepath <- path.expand(filepath)
  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }
  
  ptr <- load_model_cpp(filepath, format)
  structure(list(ptr = ptr), class = "lignum_model")
}


#' Predict values for the given input data.
#'
#' @title Predict using Lignum Model
#' @description 
#' Computes predictions using a compiled Lignum decision tree model.
#' 
#' For optimal performance, the C++ backend requires a C-contiguous (row-major) 
#' memory layout. Since R matrices are Fortran-contiguous (column-major) by default, 
#' an internal transpose operation is triggered to ensure maximum cache efficiency.
#'
#' @param object A \code{lignum_model} object returned by \code{\link{load_model}}.
#' @param X A numeric matrix of input samples, shape \code{(n_samples, n_features)}.
#' @param raw_score Whether to predict raw scores.
#' @param n_jobs Integer specifying the number of OpenMP threads to run. 
#'   \code{-1} means using all available cores. Default is \code{-1}.
#'
#' @return A numeric vector of shape \code{(n_samples,)} containing the predicted values.
#' @export
#'
#' @examplesIf FALSE
#' X <- matrix(rnorm(1000 * 50), nrow = 1000, ncol = 50)
#' preds <- predict(model, X, n_jobs = -1L)
predict.lignum_model <- function(object, X, raw_score = FALSE, n_jobs = -1L, ...) {
  if (!inherits(object, "lignum_model")) {
    stop("Object is not a valid Lignum model.")
  }
  
  if (!is.matrix(X) || typeof(X) != "double") {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
  }
  
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  X_flat <- as.vector(t(X))
  
  preds <- predict_cpp(object$ptr, X_flat, as.integer(n_samples), as.integer(n_features), as.logical(raw_score), as.integer(n_jobs))
  return(preds)
}


#' Save the model to Lignum binary format.
#'
#' @title Save Lignum Model
#' @description
#' Saves the compiled model to an binary format for mmap-based loading.
#'
#' @param object A \code{lignum_model} object.
#' @param filepath A string specifying the destination path for the \code{.bin} file.
#'
#' @export
#'
#' @examplesIf FALSE
#' save_model(model, "fast_model.bin")
save_model <- function(object, filepath) {
  if (!inherits(object, "lignum_model")) {
    stop("Object is not a valid Lignum model.")
  }
  
  filepath <- path.expand(filepath)
  save_model_cpp(object$ptr, filepath)
  invisible(NULL)
}