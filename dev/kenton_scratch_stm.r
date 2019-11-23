library(govtrackR)
library(dplyr)
library(stm)

# get some data crudely with "artificial intelligence"
fpds_ai <-
  fpds_free_text_search("artificial intelligence", parse_contracts = TRUE)

fpds_ai %>% filter(is.na(yearObligation)) %>%
  select(matches("date"), descriptionContractRequirement, urlFPDSContract, urlFPDSContractAtom) %>% View()

df_by_year <-
  fpds_ai %>%
  group_by(yearContractEffective) %>%
  summarize(count = n()) %>%
  ungroup()

df_by_year %>%
  {
    plot(count ~ yearContractEffective, data = ., type = "l")
  }



fpds_ai %>%
  group_by(year, nameVendor) %>%
  summarize(count = n()) %>%
  top_n(10, count) %>%
  ungroup() %>%
  arrange(desc(year), desc(count))

processed <- textProcessor(fpds_ai$descriptionContractRequirement, metadata = fpds_ai)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs <- out$documents
vocab <- out$vocab
meta <-out$meta

# not really all that relevant since only modelling by year but we can decide on other meta to  evaluate
yearFit <- stm(
  documents = out$documents,
  vocab = out$vocab,
  K = 20, prevalence =~ year,
  max.em.its = 75, data = out$meta,
  init.type = "Spectral"
)

plot(yearFit, type = "summary", xlim = c(0, .3))
plot(yearFit, type = "perspectives", topics = c(4, 15))
plot(yearFit, type="hist")
topicQuality(model=yearFit, documents=docs)
# devtools::install_github("timelyportfolio/stmBrowser@htmlwidget")
stmBrowser::stmBrowser_widget(
  yearFit,
  data = meta,
  "year",
  text = "descriptionContractRequirement"
)