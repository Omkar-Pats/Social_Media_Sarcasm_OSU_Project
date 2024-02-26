setwd('C:\\__\\Project\\python_ice_and_tha\\project-deliverable-1-cia')

library(tidyverse)
library(tidytext)
library(dplyr)
library(tm)
#Stemming packages
library(SnowballC)

olddf = read.csv('data\\base_data\\base-data-sarcasm.csv', header = T)
head(olddf)
df = olddf[olddf$label == 1, ]
summary(df)
comments = select(df,comment)
tidy_dataset = unnest_tokens(comments, word, comment, token = 'words')
counts = count(tidy_dataset, word)

result1 = arrange(counts, desc(n))
head(result1)

##removing stopwords
data("stop_words")
tidy_dataset2 = anti_join(tidy_dataset, stop_words, copy = T, by = 'word')
counts2 = count(tidy_dataset2, word)

result2 = arrange(counts2, desc(n))
head(result2)

##removing digits
patterndigits = '\\b[0-9]+\\b'
tidy_dataset2$word = str_remove_all(tidy_dataset2$word, patterndigits)
counts3 = count(tidy_dataset2, word)
result3 = arrange(counts3, desc(n))
head(result3)

##removing spaces,tabs and new lines
tidy_dataset2$word = str_replace_all(tidy_dataset2$word, '[:space:]', '')
tidy_dataset3 = filter(tidy_dataset2,!(word == ''))
counts4 = count(tidy_dataset3, word)
result4 = arrange(counts4, desc(n))

list_filter = c("bad","forgot","even","every","yes","yeah","well")
tidy_dataset3 = filter(tidy_dataset3, !(word %in% list_filter))

frequency = tidy_dataset3 %>%
  count(word) %>%
  arrange(desc(n)) %>%
  mutate(proportion = (n / sum(n)*100)) %>%
  filter(proportion >= 0.2)

library(scales)

ggplot(frequency, aes(x = proportion, y = word)) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  theme(legend.position="none") +
  labs(y = 'Word', x = 'Proportion')

ggplot(frequency, aes(x = proportion, y = word)) + labs(title = 'Proportion of words in Sarcastic comments', y = 'Word', x = 'Proportion') + 
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  geom_jitter(alpha = 0.1, size = 1.5) + 
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



### for parent_comment
comments = select(df,parent_comment)
head(comments)
tidy_dataset = unnest_tokens(comments, word, parent_comment, token = 'words')
counts = count(tidy_dataset, word)

result1 = arrange(counts, desc(n))
head(result1)

##removing stopwords
data("stop_words")
tidy_dataset2 = anti_join(tidy_dataset, stop_words, copy = T, by = 'word')
counts2 = count(tidy_dataset2, word)

result2 = arrange(counts2, desc(n))
head(result2)

##removing digits
patterndigits = '\\b[0-9]+\\b'
tidy_dataset2$word = str_remove_all(tidy_dataset2$word, patterndigits)
counts3 = count(tidy_dataset2, word)
result3 = arrange(counts3, desc(n))
head(result3)

##removing spaces,tabs and new lines
tidy_dataset2$word = str_replace_all(tidy_dataset2$word, '[:space:]', '')
tidy_dataset3 = filter(tidy_dataset2,!(word == ''))
counts4 = count(tidy_dataset3, word)
result4 = arrange(counts4, desc(n))

list_filter = c("bad","forgot","even","every","yes","yeah","well")
tidy_dataset3 = filter(tidy_dataset3, !(word %in% list_filter))

frequency = tidy_dataset3 %>%
  count(word) %>%
  arrange(desc(n)) %>%
  mutate(proportion = (n / sum(n)*100)) %>%
  filter(proportion >= 0.2)

library(scales)

ggplot(frequency, aes(x = proportion, y = word)) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  theme(legend.position="none") +
  labs(y = 'Word', x = 'Proportion')

ggplot(frequency, aes(x = proportion, y = word)) + labs(title = 'Proportion of words in Sarcastic parent comments', y = 'Word', x = 'Proportion') + 
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  geom_jitter(alpha = 0.1, size = 1.5) + 
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
