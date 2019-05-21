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

#' Convert sci-kit learn dataset into a tibble
#'
#' @param data sci-kit learn data set
#' @param include_group if \code{TRUE} appends group index to tibble and if group is numeric strats index at 1 instead of zero
#' @param feature_names if not \code{NULL} tibble featre names
#'
#' @return \code{tibble()}
#' @export
#'
#' @examples
#' library(tidyverse)
#' library(sklearn)
#' ds <- sk_datasets()
#' n_samples <- 1500L
#' moons <- ds$make_moons(n_samples = n_samples, noise = .05)
#'
#' df_moon <-
#' moons %>%
#' sk_ds_to_tibble(include_group = T, feature_names = c("x", "y", "group"))
#'
#' df_moon %>% glimpse()

sk_ds_to_tibble <- function(data, include_group = T, feature_names = NULL) {

  ds <-
    data[[1]] %>%
    as_tibble() %>%
    suppressWarnings()

  if (include_group & length(data) > 1) {
    group <- data[[2]]
     if (as.integer(group) %>% sum(na.rm = T) > 0) {
       group <- as.numeric(group) + 1
     }
    ds <-
      ds %>%
      mutate(group)
  }

  if (length(feature_names) > 0)  {
    ds <-
      ds %>%
      setNames(feature_names[1:ncol(ds)])
  }

  ds
}
