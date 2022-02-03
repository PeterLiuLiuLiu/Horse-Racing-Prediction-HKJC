# install.packages('reticulate')
# py_install("pandas")

require("reticulate")
library(tidyverse)
setwd('C:/Users/raymond-cy.liu/.spyder-py3/20200316 Horse racing prediction/')

source_python("R_pickle.py")
df <- read_pickle_file("C:/Users/raymond-cy.liu/.spyder-py3/20200316 Horse racing prediction/feature_df.pickle")


Win_horse_plot <- function(df, plc, type, x_lower, x_upper)
{
  win <- df %>%
    filter(Plc. <= plc) %>%
    select(type)
  all <- df %>%
    filter(Plc. > plc) %>%
    select(type)
  print(type)
  df_compare <- data.frame(c(rep('win', nrow(win)), rep('all', nrow(all))), rbind(win, all))
  colnames(df_compare) <- c('type', 'value')
  df_compare %>%
    ggplot(mapping = aes(x = value)) +
    geom_density(mapping = aes(colour = type), size = 1.2) +
    labs(title = paste(type, 'of', plc, 'win horses vs all horses')) +
    xlim(x_lower, x_upper)
}


Win_horse_facet_plot <- function(df, plc, type, facet, x_lower, x_upper, y_lower, y_upper)
{
  win <- df %>%
    filter(Plc. <= plc) %>%
    select(type, facet)
  all <- df %>%
    filter(Plc. > plc) %>%
    select(type, facet)
  df_compare <- data.frame(c(rep('win', nrow(win)), rep('all', nrow(all))), rbind(win, all))
  colnames(df_compare) <- c('type', 'value', 'facet')
  head(df_compare)
  df_compare %>%
    ggplot(mapping = aes(x = value)) +
    geom_density(mapping = aes(colour = type), size = 1.2) +
    labs(title = paste(type, 'of', plc, 'win horses vs all horses, facet with ', facet)) +
    facet_wrap(~facet, nrow = 2) +
    xlim(x_lower, x_upper) +
    ylim(y_lower, y_upper)
}


Win_horse_test <- function(df, plc, type, facet = 0)
{
  facet_item <- NULL
  score <- NULL
  for (i in 1:10)
  {
    if (facet == 0)
    {
      win <- df %>%
        filter(Plc. <= plc) %>%
        select(type) %>%
        sample_n(50, replace = TRUE)
      all <- df %>%
        filter(Plc. > plc) %>%
        select(type) %>%
        sample_n(50, replace = TRUE) 
      score <- c(score, t.test(win, all, var.equal = TRUE,  alternative = 'two.sided', mu = 0)$'p.value')
    }
    else
    {
      for (item in unique(df$class))
      {
        win <- df %>%
          filter(Plc. <= plc, class == item) %>%
          select(type) %>%
          sample_n(50, replace = TRUE)
        all <- df %>%
          filter(class == item) %>%
          select(type) %>%
          sample_n(50, replace = TRUE) 
        score <- c(score, t.test(win, all, var.equal = TRUE,  alternative = 'two.sided', mu = 0)$'p.value')
        facet_item <- c(facet_item, item)
      }
    }
  }
  if (facet == 0) {print(paste('Mean P-value for', type, ':', mean(score)))}
  else
  {
    score_df <- data.frame('class'= facet_item, 'score'= score)
    score_df %>% group_by(class)%>% summarise(mean = mean(score))
  }
}


Win_horse_plot(df, 1, 'Priority', -2, 15)
Win_horse_facet_plot(df, 1, 'Priority', 'Plc.', -2, 15, 0, 0.5)

Win_horse_test(df, plc = 1, type = 'New_Gear', facet = 0)

