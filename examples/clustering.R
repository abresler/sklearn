library(sklearn)
library(tidyverse)


data_sets <- sk_datasets()
metrics <- sk_metrics()
embedder <- sk_cluster()
df <-
  iris %>% select_if(is.numeric) %>% as_tibble()

km <- embedder$KMeans(n_clusters = 3L, verbose = 1L, init = 'random')
model <- km$fit(X = df)

df_km <-
  model$fit_transform(X = df) %>%
  sk_tibble(feature_names = NULL) %>%
  mutate(Species = as.character(iris$Species))

df_km %>%
  ggplot(aes(V1, V2, color = Species)) +
  geom_jitter() +
  ggtitle("KMeans Clusters") +
  theme_minimal()
