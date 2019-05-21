library(tidyverse)
library(sklearn)
library(skimr)

sk_ds <- sk_datasets()

# california housing ------------------------------------------------------
ca_housing <- sk_ds$california_housing
data_ca_housing <- ca_housing$fetch_california_housing()
data_ca_housing$DESCR
target <- data_ca_housing$target
ca_names <- data_ca_housing$feature_names
df_ca_housing <- data_ca_housing$data %>% sk_tibble(feature_names = ca_names)
df_ca_housing <- df_ca_housing %>%
  mutate(priceMedian = target)
