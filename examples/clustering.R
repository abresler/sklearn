library(sklearn)
library(tidyverse)


data_sets <- sk_datasets()
metrics <- sk_metrics()
embedder <- sk_cluster()

df <-
  iris %>% select_if(is.numeric) %>% as_tibble()


# kmeans ------------------------------------------------------------------
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


# Agglomerative Cluster ---------------------------------------------------
AggClust <-
  embedder$AgglomerativeClustering(n_clusters = 3L, affinity = "euclidean")
model <- AggClust$fit(X = df)
cluster <- model$fit_predict(X = df) + 1
cluster <- glue::glue("Cluster {cluster}") %>% as.character()
embedder$SpectralClustering()
## Basic Plot
df %>%
  mutate(cluster) %>%
  ggplot(aes(Sepal.Length, Sepal.Width, color = cluster)) +
  geom_jitter() +
  ggtitle("Agglomerative Clusters - Euclidean Distance") +
  theme_minimal()

manifold <- sk_manifold()
spectral_embedding <-
  manifold$SpectralEmbedding(n_components = 2L)

model_spectral <- spectral_embedding$fit(X = df)

df_spectral <-
  model_spectral$fit_transform(X = scale(df)) %>%
  sk_tibble(feature_names = c("spectral1", "spectral2")) %>%
  mutate(cluster)

df_spectral %>%
  mutate(cluster) %>%
  ggplot(aes(spectral1, spectral2, color = cluster)) +
  geom_jitter() +
  ggtitle("Agglomerative Clusters - Euclidean Distance by Spectral Cluster") +
  theme_minimal()

# cluster_examples --------------------------------------------------------

ds <- sk_datasets()
n_samples <- 1500L
moons <- ds$make_moons(n_samples = n_samples, noise = .05)
df_moon <-
  moons %>%
  sk_ds_to_tibble(include_group = T, feature_names = c("x", "y", "group"))

df_moon
