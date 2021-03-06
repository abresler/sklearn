library(sklearn)
library(tidyverse)

manifold <- sk_manifold()
data_sets <- sk_datasets()

## isomap

digits <- data_sets$load_digits(n_class = 6)
X <- digits$data %>% as.matrix()
y <- digits$target
n_neighbors <- 30L
n_features <- ncol(X) %>% as.integer()
n_samples <- nrow(X) %>% as.integer()
isomap <- manifold$Isomap(n_neighbors = n_neighbors, n_components = 2L)
model <- isomap$fit(X = X, y = y)

df <-
  model$fit_transform(X = X) %>%
  sk_tibble(c("iso1", "iso2")) %>%
  mutate(digit = glue::glue("# {y}") %>% as.character())

df %>%
  ggplot(aes(x = iso1, y= iso2, color = digit)) +
  geom_jitter(size = .25) +
  theme_minimal() +
  ggtitle("MNIST Digit Isomap Embeddings")

### Linear Embedding
linear_embedding <- manifold$LocallyLinearEmbedding(n_neighbors = n_neighbors, n_components = 2L)
model <- linear_embedding$fit(X = X, y = y)

df_linear_embed <-
  model$fit_transform(X = X) %>%
  sk_tibble(c("linear1", "linear2")) %>%
  mutate(digit = glue::glue("# {y}") %>% as.character())

df_linear_embed %>%
  ggplot(aes(x = linear1, y= linear2, color = digit)) +
  geom_jitter(size = .25) +
  theme_minimal() +
  ggtitle("MNIST Digit Linear Embeddings")

### MDS

clf <-  manifold$MDS(n_components = 2L,
                    n_init = 1L,
                    max_iter = 100L)

model <- clf$fit(X = X)
df_clf <-
  model$fit_transform(X = X) %>%
  sk_tibble(c("mds1", "mds2")) %>%
  mutate(digit = glue::glue("# {y}") %>% as.character())

df_clf %>%
  ggplot(aes(x = mds1, y= mds2, color = digit)) +
  geom_jitter(size = .25) +
  theme_minimal() +
  ggtitle("MNIST Digit MDS Embeddings")

## Spectral Clustering

embedder <-
  manifold$SpectralEmbedding(
    n_components = 2L,
    random_state = 0L,
    eigen_solver = "arpack"
  )

df_spectral <-
  embedder$fit_transform(X = X) %>%
  sk_tibble(c("spectral1", "spectral2")) %>%
  mutate(digit = glue::glue("# {y}") %>% as.character())

df_spectral %>%
  ggplot(aes(x = spectral1, y= spectral2, color = digit)) +
  geom_jitter(size = .05) +
  theme_minimal() +
  ggtitle("MNIST Digit Spectral Clustering Embeddings")

## T-SNE

tsne <- manifold$TSNE(n_components = 2L,
                     init = 'pca',
                     random_state = 0L)

df_tsne <-
  tsne$fit_transform(X = X) %>%
  sk_tibble(c("tsne1", "tsne2")) %>%
  mutate(digit = glue::glue("# {y}") %>% as.character())


df_tsne %>%
  ggplot(aes(x = tsne1, y= tsne2, color = digit)) +
  geom_jitter(size = .05) +
  theme_minimal() +
  ggtitle("MNIST Digit T-SNE Embeddings")
