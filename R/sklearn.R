# https://scikit-learn.org/0.15/index.html

#' Import SK Learn
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#api-reference}{SK learn API from python}
#'
#' @param assign_to_environment if \code{TRUE} assings to environment
#'
#' @return python object
#' @export
#'
#' @examples
import_sklearn <-
  function(assign_to_environment = T) {
    sklearn <- reticulate::import("sklearn")
    !'sklearn' %>% exists() & assign_to_environment
    if (assign_to_environment) {
      assign('sklearn', sklearn, envir = .GlobalEnv)
    }
    sklearn
  }

#' SKLearn Clustering
#'
#' Functions for unsupervised clustering algorithms.
#'
#' \itemize{
#' \item \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster}{clustering functions}
#' \item \href{https://scikit-learn.org/stable/modules/clustering.html#clustering}{clustering examples}
#' }
#'
#' @return python object
#' @export
#'
#' @examples
sk_cluster <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$cluster
    obj
  }



#' SKLearn Data
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets}{datasets}
#'
#' @return python object
#' @export
#'
#' @examples
sk_datasets <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$datasets
    obj
  }

#' SK Learn Preprocessing
#'
#' Functions to pre-process data
#'
#'\itemize{
#'\item \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing}{preprocessing functions}
#' \item \href{https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing}{preprocessing examples}
#' }
#'
#' @return python object
#' @export
#'
#' @examples
sk_preprocessing <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$preprocessing
    obj
  }


#' SK Learn LM

#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model}{linear model functions}
#' \href{https://scikit-learn.org/stable/modules/linear_model.html#linear-model}{linear model examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_lm <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$linear_model
    obj
  }

#' Neighbors
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors}{nearest neighbor functions}
#' \href{https://scikit-learn.org/stable/modules/neighbors.html#neighbors}{nearest neighbor examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_neighbors <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$neighbors
    obj
  }

#' SK Decomposition
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition}{decomposition functions}
#' \href{https://scikit-learn.org/stable/modules/decomposition.html#decompositions}{decomposition example}
#'
#' @return python object
#' @export
#'
#' @examples
sk_decomp <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$decomposition
    obj
  }


#' SK Manifold
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold}{manifold functions}
#' \href{https://scikit-learn.org/stable/modules/manifold.html#manifold}{manifold examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_manifold <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$manifold
    obj
  }

#' SK Ensemble
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble}{ensemble functions}
#' \href{https://scikit-learn.org/stable/modules/ensemble.html#ensemble}{ensemble examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_ensemble <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$ensemble
    obj
  }

#' SK Covariance
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance}{co-variance functions}
#' \href{https://scikit-learn.org/stable/modules/covariance.html#covariance}{co-variance examples}
#' @return python object
#' @export
#'
#' @examples
sk_covariance <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$covariance
    obj
  }

#' SK Model Selection
#'
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance}{model selection functions}
#' \href{https://scikit-learn.org/stable/modules/covariance.html#covariance}{model selectionexamples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_model_selection <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$model_selection
    obj
  }

#' SK Tree Models
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree}{tree functions}
#' \href{https://scikit-learn.org/stable/modules/tree.html#tree}{tree examples}
#'
#' @return python object
#' @export
#'
#' @examples
#' library(tidyverse)
#' X <- iris %>% select_if(is.numeric) %>% as_tibble()
#' target <- iris %>% pull(Species)
#' tree <- sk_tree()
#' tree_classifier <- tree$DecisionTreeClassifier()
#' model <- tree_classifier$fit(X = X, y = target)
#' model$predict(X = X)
sk_tree <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$tree
    obj
  }

#' SK Utilities
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils}{utility functions}
#'
#' @return python object
#' @export
#'
#' @examples
sk_utils <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$utils
    obj
  }

#' SK SVM Models
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm}}{SVM functions}
#' \href{https://scikit-learn.org/stable/modules/svm.html#svm}{SVM examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_svm <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$svm
    obj
  }


#' SK Bayes
#' These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.
#'
#' \itemize{
#' \item \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes}{Bayes functions}
#' \item \href{https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes}{Bayes examples}
#' }
#'
#' @return python object
#' @export
#'
#' @examples
#' library(tidyverse)
#' library(sklearn)
#' library(caret)
#' bayes <- sk_bayes()
#' X <- iris %>% as_tibble %>% select_if(is.numeric)
#' target <- iris %>% pull(Species)
#' gnb <- bayes$GaussianNB()
#' model <- gnb$fit(X = as.matrix(X), y = target)
#' predictions <- model$predict(X = X)
#' result_table <- table(target, predictions)
#' confusionMatrix(result_table)
#' ### Class probability
#' options(digits = 3, scipen = 9999)
#' df_prob <-
#' gnb$predict_proba(X = X) %>%
#' as_tibble() %>%
#' set_names(target %>% unique())
#'
#' df_prob

sk_bayes <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <- sklearn$naive_bayes
    obj
  }

#' SK Calibration
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration}{Calibration functions}
#' \href{https://scikit-learn.org/stable/modules/calibration.html#calibration}{Calibration examples}
#'
#'
#' @return python object
#' @export
#'
#' @examples
sk_calibration <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$calibration
    obj
  }

#' SK Cross Decomposition
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition}{Cross-decomposition functions}
#' \href{https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition}{Cross decomposition examples}
#' @return python object
#' @export
#'
#' @examples
sk_cross_decomposition <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$cross_decomposition
    obj
  }

#' SK Dummy Variables
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy}{Dummy variable functions}
#' @return python object
#' @export
#'
#' @examples
sk_dummy <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$dummy
    obj
  }

#' SK Discriminant Models
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis}{Discriminant functions}
#' \href{https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda}{Discriminant examples}
#'
#' @return python object
#' @export
#'
#' @examples
sk_discriminant_analysis <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$discriminant_analysis
    obj
  }


#' SK Feature Selection
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection}{Feature selection functions}
#' \href{https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection}{feature selection examples}
#' @return python object
#' @export
#'
#' @examples
sk_feature_selection <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$feature_selection
    obj
  }

#' SK Neural Net
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network}{Neural Network functions}
#' \href{https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised}{Neural Network examples}
#' @return python object
#' @export
#'
#' @examples
sk_nnet <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$neural_network
    obj
  }

#' SK Gaussian Process
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process}{Gaussian process functions}
#' \href{https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process}{Gaussian process examples}
#'
#' @return python jobject
#' @export
#'
#' @examples
sk_gaussian_process <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$gaussian_process
    obj
  }

#' SK Mixture
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture}{Mixture functions}
#' @return python object
#' @export
#'
#' @examples
sk_mixture <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$mixture
    obj
  }


#' SK Isotonic
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic}{Isotonic functions}
#' \href{https://scikit-learn.org/stable/modules/isotonic.html#isotonic}{Isotonic examples}
#' @return python object
#' @export
#'
#' @examples
sk_isotonic <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$isotonic
    obj
  }
#' SK Impute
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute}{impute functions}
#' @return python object
#' @export
#'
#' @examples
sk_impute <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$impute
    obj
  }
#' SK Kernel Approxomation
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation}{Kernel Approxomation functions}
#' \href{https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation}{Kernel Approxomation examples}
#' @return python object
#' @export
#'
#' @examples
sk_kernel_approximation  <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$kernel_approximation
    obj
  }
#' SK Kernel Ridge
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge}{Kernel Ridge functions}
#' \href{https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge}{Kernel Ridge examples}
#' @return python object
#' @export
#'
#' @examples
sk_kernel_ridge <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$kernel_ridge
    obj
  }
#' SK Metrics
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics}{metrics functions}
#' @return python object
#' @export
#'
#' @examples
sk_metrics <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$metrics
    obj
  }
#' SK Multi Class
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass}{multi-class functions}
#' \href{https://scikit-learn.org/stable/modules/multiclass.html#multiclass}{multiclass examples}
#' @return
#' @export
#'
#' @examples
sk_multi_class <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$multi_class
    obj
  }
#' SK Multioutput
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput}{Multioutput functions}
#' \href{https://scikit-learn.org/stable/modules/multiclass.html#multiclass}{multioutput examples}
#' @return python object
#' @export
#'
#' @examples
sk_multioutput <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$multioutput
    obj
  }

#' SK Random Projection
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection}{Random projection functions}
#' \href{https://scikit-learn.org/stable/modules/random_projection.html#random-projection}{Random Projection examples}
#' @return python object
#' @export
#'
#' @examples
sk_random_projection <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$random_projection
    obj
  }

#' SK semi-supervised
#' \href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised}semi-supervised functions}
#'
#' @return python object
#' @export
#'
#' @examples
sk_semi_supervised <-
  function() {
    sklearn <- import_sklearn(assign_to_environment = F)
    obj <-
      sklearn$semi_supervised
    obj
  }