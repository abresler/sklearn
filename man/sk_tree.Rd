% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sklearn.R
\name{sk_tree}
\alias{sk_tree}
\title{SK Tree Models
\href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree}{tree functions}
\href{https://scikit-learn.org/stable/modules/tree.html#tree}{tree examples}}
\usage{
sk_tree()
}
\value{
python object
}
\description{
SK Tree Models
\href{https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree}{tree functions}
\href{https://scikit-learn.org/stable/modules/tree.html#tree}{tree examples}
}
\examples{
library(tidyverse)
X <- iris \%>\% select_if(is.numeric) \%>\% as_tibble()
target <- iris \%>\% pull(Species)
tree <- sk_tree()
tree_classifier <- tree$DecisionTreeClassifier()
model <- tree_classifier$fit(X = X, y = target)
model$predict(X = X)
}
