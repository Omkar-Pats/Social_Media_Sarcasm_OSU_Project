setwd("C:\\__\\Project\\python_ice_and_tha\\project-deliverable-1-cia\\data\\vectorized_text_data")

library(ggplot2)
library(tidyr)

# for vectorized_comments
df = read.csv("comments_data_vectorized.csv", header = T)
head(df)

df2 <- subset(df, select = -c(0,1,2,3))
head(df2)
colMeans(df2)
# calculate the average values of each column
df_avg <- data.frame(variable = colnames(df2)[-1], average = colMeans(df2[-1]))

# reshape the data into a "long" format
df_avg_long <- gather(df_avg, key = "variable", value = "average")
head(df_avg_long)

library(dplyr)
sorted_avg_df <- arrange(df_avg_long, desc(average))
head(sorted_avg_df,10)
top_10_means_df <- head(sorted_avg_df, n = 10)
top_10_means_df
# plot the average values using ggplot
ggplot(top_10_means_df, aes(x = variable, y = average)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average tfidf values of each word", x = "Word", y = "Average value") + 
  theme(
    # setting up axis formatting
    text = element_text(size=12, family = 'Helvetica', color = 'white'),
    axis.title = element_text(size=15, family = 'Helvetica', color = 'white'),
    axis.text.x = element_text(size = 12, family = 'Arial', color = 'white'),
    axis.text.y = element_text(size = 12, family = 'Arial', color = 'white'),
    axis.ticks = element_line(colour = "lightgrey"),
    axis.line = element_line(colour = "lightgrey"),
    
    # setting up background formatting
    plot.background = element_rect(fill = "gray60"),
    panel.background = element_rect(fill = "white"),
    panel.grid= element_line(color = "gray90", size = 0.5, linetype = 2),
    
    # setting up title and subtitle formatting
    plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
    plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic')
  )



# for parent posts
df = read.csv("parent_comments_data_vectorized.csv", header = T)
head(df)

df2 <- subset(df, select = -c(0,1,2,3))
head(df2)
colMeans(df2)
# calculate the average values of each column
df_avg <- data.frame(variable = colnames(df2)[-1], average = colMeans(df2[-1]))

# reshape the data into a "long" format
df_avg_long <- gather(df_avg, key = "variable", value = "average")
head(df_avg_long)

library(dplyr)
sorted_avg_df <- arrange(df_avg_long, desc(average))
head(sorted_avg_df,10)
top_10_means_df <- head(sorted_avg_df, n = 10)
top_10_means_df
# plot the average values using ggplot
ggplot(top_10_means_df, aes(x = variable, y = average)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average tfidf values of each word in parent comment", x = "Word", y = "Average value") +
  theme(
    # setting up axis formatting
    text = element_text(size=12, family = 'Helvetica', color = 'white'),
    axis.title = element_text(size=15, family = 'Helvetica', color = 'white'),
    axis.text.x = element_text(size = 12, family = 'Arial', color = 'white'),
    axis.text.y = element_text(size = 12, family = 'Arial', color = 'white'),
    axis.ticks = element_line(colour = "lightgrey"),
    axis.line = element_line(colour = "lightgrey"),
    
    # setting up background formatting
    plot.background = element_rect(fill = "gray60"),
    panel.background = element_rect(fill = "white"),
    panel.grid= element_line(color = "gray90", size = 0.5, linetype = 2),
    
    # setting up title and subtitle formatting
    plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
    plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic')
  )

