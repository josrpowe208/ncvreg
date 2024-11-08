library(tidyverse)
source("/Users/jrp208/Documents/CWRU/Li_lab/ncvreg/src/R/ncvreg/R/ncvreg.R")

# Load data
# Pick 4 random csv files from a directory and load then into 1 dataframe
df <- data.frame()
files <- list.files(path = "/Users/jrp208/Documents/Independent_Work/data/kaggle_stocks/stocks", pattern = "\\.csv$", full.names = TRUE)

# Randomly select n files from file
idxs <- sample(1:length(files), 4, replace = FALSE)
for (idx in idxs) {
    file <- files[idx]
    temp <- read_csv(file)
    names(temp) <- tolower(names(temp))
    names(temp) <- str_replace_all(names(temp), " ", "_")
    # Add stock name to temp
    symbol <- str_split(str_split(file, "/")[-1], "\\.")[1]
    temp <- temp %>% mutate(symbol = symbol)

    df <- bind_rows(df, temp)
}

X <- df %>% select(-date, -symbol, -adj_close) %>% as.matrix()
y <- df$adj_close %>% as.matrix()

model <- ncvreg(X, y, penalty = "lasso")


