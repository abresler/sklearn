## pip3 install scikit-learn
## pip3 install umap
library(sklearn) # remotes::install_github("abresler/sklearn")
library(tidyverse)
library(uwot)
library(asbmisc)
library(reticulate)
## https://gist.github.com/lmcinnes/fbb63592b3225678390f08e50eda2b61#file-20newsgroups-docmap-example-ipynb
skd <- sk_datasets()
news <- skd$fetch_20newsgroups(subset = 'all')
sk <- import_sklearn()
skf <- sk_feature_extraction()

vectorizer  <-
  skf$text$CountVectorizer(min_df = 10L, stop_words = 'english')


count_data =
  vectorizer$fit_transform(raw_documents = as.character(news$data))

count_data
umap  <- import("umap")
mapper = umap$UMAP(metric = "euclidean")
d <- mapper$fit(X = count_data)
df_ids <- tibble(name = news$target_names,
                 id = news$target %>% unique() %>% sort())
count_data %>% str

df_umap <- d$embedding_ %>%
  as_tibble() %>%
  set_names(c("umap1", "umap2")) %>%
  mutate(id = news$target) %>%
  left_join(df_ids, by = "id")


df_umap %>%
  ggplot(aes(umap1, umap2, color = name)) +
  geom_jitter(size = .25) +
  hrbrthemes::theme_ipsum()

df_umap %>%
  sample_n(20) %>%
  hc_xy(x = "umap1",
        y = "umap2",
        title = "News20 UMAP",
        type = "euclidean distance -- in R")

df_umap %>%
  rename(g = name) %>%
  hc_xy(
    x = "umap1",
    y = "umap2",
    group = "g",
    point_size = 1.75,
    point_width = 5,
    title = "News20 UMAP",
    theme_name = "ft"
  )
