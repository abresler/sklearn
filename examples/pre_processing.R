library(sklearn)
library(tidyverse)

df <- iris %>% select_if(is.numeric) %>% as_tibble()
scaler <- sk_preprocessing()

df_min_max <-
  scaler$minmax_scale(X = df) %>%
  as_tibble() %>%
  set_names(names(df))

scaler$KernelCenterer$fit()
