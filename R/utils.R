#' Convert SK Matrix to Tibble
#'
#' @param data SK data output
#' @param feature_names if not \code{NULL} data feature names
#'
#' @return \code{tibble}
#' @export
#'
#' @examples
sk_tibble <-
  function(data, feature_names = NULL) {
    data <- as_tibble(data)

    if (length(feature_names) > 0) {
      data <- set_names(data, feature_names)
    }
    data
  }
