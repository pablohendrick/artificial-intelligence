library(ggplot2)      
library(dplyr)        
library(tidyr)        
library(magrittr)     
library(plotly)       
library(VIM)          
library(missForest)   

options(warn=-1)  

htmlwidgets::initNotebook()

data <- read.csv('../input/world-population-dataset/world_population.csv')

head(data)

num_rows <- nrow(data)
num_cols <- ncol(data)

print(num_rows)
print(num_cols)

summary_stats <- summary(data)

summary_stats <- t(summary_stats)

summary_stats <- summary_stats[order(summary_stats$mean, decreasing = TRUE), ]

library(scales)

heatmap_color <- scales::col_numeric(palette = "BuGn")(summary_stats)

library(ggplot2)
library(dplyr)
library(tidyr)

summary_stats %>%
  mutate_if(is.numeric, funs(as.numeric(as.character(.)))) %>%
  gather(key = "Statistic", value = "Value") %>%
  mutate(Color = ifelse(Statistic == "std", "red", "blue")) %>%
  ggplot(aes(x = reorder(rownames(summary_stats), -mean), y = Value, fill = Color)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_identity() +
  labs(title = "Descriptive Statistics Ordered by Mean") +
  theme_minimal()

features <- names(data)
for (feature in features) {
    unique_values <- unique(data[[feature]])
    cat(paste(feature, "--->", length(unique_values), "\n"))
}

missing_values <- colSums(is.na(data))

print(missing_values)

aggr_plot <- aggr(data, col=c('blue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3)

plot(aggr_plot)

continent_data <- data %>%
  group_by(Continent) %>%
  summarise(across(all_of(features), mean)) %>%
  arrange(desc(`Density (per km²)`))

library(scales)

heatmap_color <- scales::col_numeric(palette = "BuGn")(continent_data)

heatmap_color

continent_data <- continent_data[order(continent_data$`2022 Population`), ]

population_features <- c('2022 Population', '2020 Population', '2015 Population', '2010 Population',
                         '2000 Population', '1990 Population', '1980 Population', '1970 Population')
population_features <- population_features[length(population_features):1]

library(ggplot2)

library(tidyr)
continent_data_long <- gather(continent_data, key = "Year", value = "Population", -Continent)

ggplot(data = continent_data_long, aes(x = as.factor(Continent), y = Population, group = Year, color = Year)) +
  geom_line() +
  labs(x = "Continent", y = "Population", title = "Population Over Time by Continent") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

library(plotly)

fig <- plot_geo(data, locationmode = 'country names', color = ~`2022 Population`) %>%
  add_markers(locations = ~`Country/Territory`, text = ~`Country/Territory`) %>%
  colorbar(title = '2022 Population') %>%
  layout(title = '2022 Population', geo = list(showframe = FALSE, projection = list(type = 'mercator')))

fig



