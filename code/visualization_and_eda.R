###################################
# - Calling necessary libraries
###################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(extrafont)
library(reshape2)
library(stringr)
library(forcats)

# Setting up working directory
setwd("C:\\Users\\roedj\\Documents\\GitHub\\homework_MSIS5223\\project-deliverable-1-cia")

# Creating folder to store visuals
if(!dir.exists("assets\\visualization")) {
  # create folder
  dir.create("assets\\visualization")
}

###################################
# - Data Preprocessing
###################################

#########################
#######-base_data-#######

# Read dataset
base_df = read.csv("data\\base_data\\base-data-sarcasm.csv", header = T)

summary(base_df)
str(base_df)

# Converting data types
base_df$subreddit = as.factor(base_df$subreddit)
base_df$id = as.character(base_df$id)
base_df$created_utc = as.POSIXct(base_df$created_utc, format = "%Y-%m-%d %H:%M:%S")


#########################
#######-user_data-#######

# Read dataset
user_df = read.csv("data\\user_data\\user_info.csv", header = T)

summary(user_df)
str(user_df)

# Converting data types
user_df$top_gilded_subreddit = as.factor(user_df$top_gilded_subreddit)
user_df$join_date = as.POSIXct(user_df$join_date, format = "%Y-%m-%d")

#########################
#######-join_data-#######

df = merge(base_df, user_df, by = "author")


str(df)
colnames(df)

###################################
# - Univariate Analysis
###################################

#########################
#######-base_data-#######

# Select only the numeric columns in base_df
base_num <- dplyr::select_if(base_df, is.numeric)

# find mean, median, standard deviation, maximum, and minimum values
summary_df <- data.frame(
  mean = apply(base_num, 2, mean),
  median = apply(base_num, 2, median),
  sd = apply(base_num, 2, sd),
  max = apply(base_num, 2, max),
  min = apply(base_num, 2, min)
)

# Print the summary statistics
print(summary_df)

# Loop through each column and create a histogram
for(col in names(base_num)) {
  if (n_distinct(base_num[[col]]) < 3){
    base_num[[col]] = as.character(base_num[[col]])
  }
  plot_theme = ( 
    ggplot(base_num, aes(x = .data[[col]])) +
      # setting up x and y axis label, title, and subtitle
      labs(
        x = paste(col), 
        y = "Frequency",
        title = paste("Histogram of", col)
      ) +
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
        panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
        
        # setting up title and subtitle formatting
        plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
        plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic')
      ))
  if (n_distinct(base_num[[col]]) < 3){
    plot_uni = plot_theme + geom_bar(stat = 'count') 
    ggsave(filename = paste("assets\\visualization\\univariate_", col, ".png", sep = ''), plot = plot_uni, width = 10, height = 6, dpi = 300)
  } else {
    plot_uni = plot_theme + geom_histogram() 
    ggsave(filename = paste("assets\\visualization\\univariate_", col, ".png", sep = ''), plot = plot_uni, width = 10, height = 6, dpi = 300)
  }
}

#########################
#######-user_data-#######

user_num = df %>% 
  group_by(author) %>% 
  summarise(sarcasm_sum = sum(label),
            total = n(),
            post_karma = mean(post_karma),
            comment_karma = mean(comment_karma),
            gilded_posts = mean(gilded_posts),
            gilded_unique_subs_count = mean(gilded_unique_subs_count)
  ) %>% 
  mutate(sarcasm_rate = sarcasm_sum/total) %>%
  select(post_karma, comment_karma, gilded_posts, gilded_unique_subs_count)

# find mean, median, standard deviation, maximum, and minimum values
summary_df <- data.frame(
  mean = apply(user_num, 2, mean),
  median = apply(user_num, 2, median),
  sd = apply(user_num, 2, sd),
  max = apply(user_num, 2, max),
  min = apply(user_num, 2, min)
)

# Print the summary statistics
print(summary_df)

# Loop through each column and create a histogram
for(col in names(user_num)) {
  if (n_distinct(user_num[[col]]) < 3){
    user_num[[col]] = as.character(user_num[[col]])
  }
  plot_theme = ( 
    ggplot(user_num, aes(x = .data[[col]])) +
      # setting up x and y axis label, title, and subtitle
      labs(
        x = paste(col), 
        y = "Frequency",
        title = paste("Histogram of", col)
      ) +
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
        panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
        
        # setting up title and subtitle formatting
        plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
        plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic')
      ))
  if (n_distinct(user_num[[col]]) < 3){
    plot_uni = plot_theme + geom_bar(stat = 'count') 
    ggsave(filename = paste("assets\\visualization\\univariate_", col, ".png", sep = ''), plot = plot_uni, width = 10, height = 6, dpi = 300)
  } else {
    plot_uni = plot_theme + geom_histogram() 
    ggsave(filename = paste("assets\\visualization\\univariate_", col, ".png", sep = ''), plot = plot_uni, width = 10, height = 6, dpi = 300)
  }
}

###################################
# - Multivariate Analysis
###################################

###################################
# - Label vs. Posted Time
###################################

# Processing time data
df$post_year = format(df$created_utc, '%Y')
df$post_month = format(df$created_utc, '%m')

# Aggregating sarcasm sum and rate
df_viz1 = df %>% 
  group_by(post_year, post_month) %>% 
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  drop_na(post_year) %>%
  mutate(sarcasm_rate = sarcasm_sum/total)

# Simple visual inspection
df_viz1_ts = ts(df_viz1$sarcasm_rate, frequency = 12, start = c(2009, 1))

plot(df_viz1_ts)

# Setting up theme
plot_theme = (
  ggplot(df_viz1, aes(x = as.Date(paste(post_year, post_month, "01", sep = "-")), group = 1)) +
    # setting up x and y axis lable, title, and subtitle
    labs(
      x = "Year", 
      y = "Sarcasm Count and Rate",
      title = 'Sarcasm Rate and Count on Reddit from 2009 to 2016',
      subtitle = 'To investigate the association between sarcasm on Reddit and time the post/comment created.'
    ) +
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
      panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
      
      # setting up title and subtitle formatting
      plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
      plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic'),
      
      # setting up legend formatting
      legend.text = element_text(size=14, family = 'Helvetica', color = 'white'),
      legend.title = element_text(size=14, family = 'Georgia', color = 'white', face='bold'),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "gray60"),
      legend.key = element_rect(fill = "gray60", colour = NA)
    ) 
)

# instantiate parameters for arrow
arrow_1 = tibble(
  x1 = c(as.Date(paste("2012", "01", "01", sep = "-"))),
  y1 = c(1550),
  x2 = c(as.Date(paste("2012", "05", "01", sep = "-"))),
  y2 = c(1200)
)

arrow_2 = tibble(
  x1 = c(as.Date(paste("2012", "05", "01", sep = "-"))),
  y1 = c(550),
  x2 = c(as.Date(paste("2014", "05", "01", sep = "-"))),
  y2 = c(450)
)

# Create plot
plot1 = plot_theme +
  geom_line(aes(y = sarcasm_sum, color = "Sarcasm Count")) +
  geom_line(aes(y = sarcasm_rate * max(df_viz1$sarcasm_sum), color = "Sarcasm Rate")) +
  scale_color_manual(name = NULL, values = c("Sarcasm Count" = "blue", "Sarcasm Rate" = "red")) +
  scale_y_continuous(name = "Sarcasm Count",
                     sec.axis = sec_axis(~./max(df_viz1$sarcasm_sum), name = "Sarcasm Rate")) +
  geom_label(x = as.Date(paste("2012", "01", "01", sep = "-")), y = 1600, size = 4, color='black', fill = 'grey',
             label = glue::glue('Sarcasm rate dropped slowly over time.')) +
  geom_label(x = as.Date(paste("2012", "05", "01", sep = "-")), y = 600, size = 4, color='black', fill = 'grey',
             label = glue::glue('Sarcasm count increased exponentially over time.')) +
  # input arrow
  geom_curve(data = arrow_1, 
             aes(x = x1, y = y1, xend = x2, yend = y2),
             arrow = arrow(length = unit(0.07, 'inch')), 
             size = 0.4,
             color = 'red', 
             curvature = -0.2) +
  geom_curve(data = arrow_2, 
             aes(x = x1, y = y1, xend = x2, yend = y2),
             arrow = arrow(length = unit(0.07, 'inch')), 
             size = 0.4,
             color = 'blue', 
             curvature = 0.2)

# Save plot
ggsave(filename = "assets\\visualization\\multivariate_01.png", plot = plot1, width = 10, height = 6, dpi = 300)


# other visualizations related to time
# Create sample data
df_viz1_ts <- data.frame(Jan = c(16, 23, 52, 90, 181, 275, 598, 1066),
                         Feb = c(4, 21, 56, 100, 184, 240, 574, 949),
                         Mar = c(13, 31, 68, 85, 184, 312, 514, 1217),
                         Apr = c(6, 39, 64, 108, 241, 293, 515, 1081),
                         May = c(9, 33, 57, 97, 277, 263, 583, 1197),
                         Jun = c(14, 34, 61, 127, 247, 310, 572, 1408),
                         Jul = c(12, 38, 95, 125, 216, 486, 699, 1685),
                         Aug = c(19, 33, 89, 172, 271, 522, 728, 1476),
                         Sep = c(9, 50, 81, 149, 231, 406, 733, 1506),
                         Oct = c(17, 43, 71, 169, 260, 434, 811, 1751),
                         Nov = c(22, 48, 72, 181, 217, 430, 927, 2032),
                         Dec = c(35, 37, 83, 188, 249, 478, 924, 1776))

# Convert data to long format
df_viz1_ts_long <- pivot_longer(df_viz1_ts, 
                                cols = everything(), 
                                names_to = "Month", 
                                values_to = "Count")

# Add a Year column to the long format data
df_viz1_ts_long$Year <- factor(rep(2009:2016, each = 12))

# Create a line chart
ggplot(df_viz1_ts_long, aes(x = Month, y = Count, group = Year, color = Year)) + 
  geom_line() +
  labs(title = "Counts by Month and Year", x = "Month", y = "Count")


###################################
# - Label vs. Join Time
###################################

# Processing join time data
df$diff_years <- round(as.numeric(difftime(format(df$created_utc, "%Y-%m-%d"), df$join_date, units = "days")) / 365.25, 1)

# Aggregating sarcasm sum and rate
df_viz2 = df %>% 
  filter(diff_years >= 0 & diff_years <=10) %>%
  group_by(diff_years) %>% 
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  drop_na(diff_years) %>%
  mutate(sarcasm_rate = sarcasm_sum/total)

# Setting up theme
plot_theme = (
  ggplot(df_viz2, aes(x = diff_years)) +
    # limiting x and y
    scale_x_continuous(breaks = seq(0,10, by = 1)) +
    # setting up x and y axis lable, title, and subtitle
    labs(
      x = "Years as Reddit User", 
      y = "Sarcasm Count and Rate",
      title = "Sarcasm Rate and Count on Reddit by User Tenure",
      subtitle = 'To investigate the association between sarcasm on Reddit and user tenure.'
    ) +
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
      panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
      
      # setting up title and subtitle formatting
      plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
      plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic'),
      
      # setting up legend formatting
      legend.text = element_text(size=14, family = 'Helvetica', color = 'white'),
      legend.title = element_text(size=14, family = 'Georgia', color = 'white', face='bold'),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "gray60"),
      legend.key = element_rect(fill = "gray60", colour = NA)
    ) 
)

# instantiate parameters for arrow
arrow_1 = tibble(
  x1 = c(5),
  y1 = c(1280),
  x2 = c(6.3),
  y2 = c(1200)
)


# Create plot
plot2 = plot_theme +
  geom_line(aes(y = sarcasm_sum, color = "Sarcasm Count")) +
  geom_line(aes(y = sarcasm_rate * max(df_viz1$sarcasm_sum), color = "Sarcasm Rate")) +
  scale_color_manual(name = NULL, values = c("Sarcasm Count" = "blue", "Sarcasm Rate" = "red")) +
  scale_y_continuous(name = "Sarcasm Count", breaks = seq(0,1500, by = 250),
                     sec.axis = sec_axis(~./max(df_viz1$sarcasm_sum), name = "Sarcasm Rate")) +
  geom_label(x = 5, y = 1350, size = 4, color='black', fill = 'grey',
             label = glue::glue('Sarcasm rate fluctuated because of\nlow data count after 6 years of tenure.')) +
  geom_label(x = 7, y = 530, size = 4, color='black', fill = 'grey',
             label = glue::glue('Sarcasm rate remained constant regardless of user tenure\nindicating no strong association between them.')) +
  # input arrow
  geom_curve(data = arrow_1, 
             aes(x = x1, y = y1, xend = x2, yend = y2),
             arrow = arrow(length = unit(0.07, 'inch')), 
             size = 0.4,
             color = 'red', 
             curvature = 0.2)

# Save plot
ggsave(filename = "assets\\visualization\\multivariate_02.png", plot = plot2, width = 10, height = 6, dpi = 300)

###################################
# - Label vs. Subreddit
###################################

# Preprocessing
df$ups <- as.numeric(df$ups)
df$downs <- as.numeric(df$downs)
df$score <- as.numeric(df$score)

# Aggregation
df_viz3 <- df %>%
  group_by(subreddit) %>%
  summarise(avg_score = mean(score),
            avg_ups = mean(ups),
            avg_downs = mean(downs),
            avg_sarcasm = mean(label))

# Setting up theme
plot_theme = (
  ggplot(df_viz3, aes(x = subreddit, y = avg_sarcasm, size = avg_score)) +
    # setting up x and y axis label, title, and subtitle
    labs(
      x = "Subreddit", 
      y = "Sarcasm Rate", 
      size = "Score",
      title = "Association between Sarcasm Rate and Subreddit",
      subtitle = 'To investigate the association between sarcasm on Reddit and subreddit along with score.'
    ) +
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
      panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
      
      # setting up title and subtitle formatting
      plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
      plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic'),
      
      # setting up legend formatting
      legend.text = element_text(size=14, family = 'Helvetica', color = 'white'),
      legend.title = element_text(size=14, family = 'Georgia', color = 'white', face='bold'),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "gray60"),
      legend.key = element_rect(fill = "gray60", colour = NA)
    ) 
)

# Create plot
plot3 = plot_theme +
  geom_point() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  scale_y_continuous(breaks = seq(0,1,0.20)) +
  ylim(c(0.2, 0.80)) +
  scale_size_continuous(range = c(5, 20)) +
  geom_label(x = "worldnews", y = 0.30, size = 4, color='black', fill = 'grey',
             label = glue::glue('Subreddit: politics & worldnews ==> Sarcastic.\n
                                Score: Low ==> Sarcastic.'))

# Save plot
ggsave(filename = "assets\\visualization\\multivariate_03.png", plot = plot3, width = 10, height = 6, dpi = 300)

###################################
# - Label vs. Post/Comment Karma
###################################

# Aggregating sarcasm sum and rate
df_viz4_1 = df %>% 
  group_by(post_karma) %>%
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  mutate(sarcasm_rate = sarcasm_sum/total)

df_viz4_2 = df %>% 
  group_by(comment_karma) %>% 
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  mutate(sarcasm_rate = sarcasm_sum/total)

# investigating skewness
hist(df_viz4_1$post_karma, breaks = 30)
hist(df_viz4_2$comment_karma, breaks = 30)
hist(log(df_viz4_1$post_karma), breaks = 30)
hist(log(df_viz4_2$comment_karma), breaks = 30)

# redo aggregation with log
df_viz4_1 = df %>% 
  filter(post_karma != 0) %>%
  mutate(log_post_karma = round(log(post_karma),2)) %>% 
  group_by(log_post_karma) %>%
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  mutate(sarcasm_rate = sarcasm_sum/total) %>%
  filter(total > 25)

df_viz4_2 = df %>% 
  filter(comment_karma != 0) %>%
  mutate(log_comment_karma = round(log(comment_karma),2)) %>% 
  group_by(log_comment_karma) %>%
  summarise(sarcasm_sum = sum(label),
            total = n()) %>% 
  mutate(sarcasm_rate = sarcasm_sum/total) %>%
  filter(total > 25)

# using lm - linear model to obtain model parameters
reg1<-lm(formula = sarcasm_rate~log_post_karma, data=df_viz4_1)  

# assigning intercept and slope for first regression line
coeff1<-coefficients(reg1)          
intercept1<-coeff1[1]
slope1 <- coeff1[2]
rsq1 <- summary(reg1)$r.squared

# same processes for second regression
reg2<-lm(formula = sarcasm_rate~log_comment_karma, data=df_viz4_2)  

coeff2<-coefficients(reg2)          
intercept2<-coeff2[1]
slope2 <- coeff2[2]
rsq2 <- summary(reg2)$r.squared

# Setting up theme
plot_theme = (
  ggplot(data = df_viz4_1, aes(y=sarcasm_rate, x=log_post_karma))+
    # setting up x and y axis label, title, and subtitle
    labs(
      x = 'Log Karma Count',
      y = 'Sarcasm Rate',
      title = "Association between Sarcasm Rate and\nUser's Post/Comment Karma",
      subtitle = "To investigate the association between sarcasm on Reddit and user's post/comment karma."
    ) +
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
      panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
      
      # setting up title and subtitle formatting
      plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
      plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic'),
      
      # setting up legend formatting
      legend.text = element_text(size=14, family = 'Helvetica', color = 'white'),
      legend.title = element_text(size=14, family = 'Georgia', color = 'white', face='bold'),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "gray60"),
      legend.key = element_rect(fill = "gray60", colour = NA)
    ) 
)

# Create Plot
plot4 = plot_theme + 
  geom_point(data = df_viz4_1, aes(y=sarcasm_rate, x=log_post_karma, color='red')) +
  geom_point(data = df_viz4_2, aes(y=sarcasm_rate, x=log_comment_karma, color='blue')) +
  scale_color_manual(values=c("red", "blue"), name="Karma", labels=c("Post", "Comment")) +
  # plot regression line 1
  geom_abline(intercept = intercept1, slope = slope1, color="red", alpha=0.5, size=1) + 
  # plot regression line 2
  geom_abline(intercept = intercept2, slope = slope2, color="blue", alpha=0.5, size=1)
# input annotation for explanation
#annotate('text', x = 8000000, y = 0.80, size = 4, color='red', 
#         label = glue::glue('Slope = ', round(slope1[[1]],9), '\nR-squared = ', round(rsq1, 6))) + 
#annotate('text', x = 7500000, y = 0.53, size = 4, color='blue', 
#         label = glue::glue('Slope = ', round(slope2[[1]],9), '\nR-squared = ', round(rsq2, 6))) +
#geom_label(x = 7000000, y = 0.25, size = 4, color='black', fill = 'grey',
#           label = glue::glue('Post Karma had stronger relationship\nwith sarcasm than Comment Karma.\nHowever, both Karmas were not significant.'))

# Save plot
ggsave(filename = "assets\\visualization\\multivariate_04.png", plot = plot4, width = 10, height = 6, dpi = 300)

###################################
# - Label vs. Gilded
###################################

# Aggregation
df_viz5 = df %>%
  group_by(top_gilded_subreddit) %>%
  summarise(avg_sarcasm = mean(label), 
            total_count = n(),
            avg_gilded_unique_subs_count = mean(gilded_unique_subs_count),
            avg_gilded_post = mean(gilded_posts)) %>%
  arrange(desc(total_count)) %>%
  filter(top_gilded_subreddit != 'None') %>%
  top_n(10, total_count) 

# Reorder levels of top_gilded_subreddit based on sorted order of data frame
df_viz5$top_gilded_subreddit <- fct_reorder(df_viz5$top_gilded_subreddit, desc(df_viz5$total_count))

# Setting up theme
plot_theme = (
  ggplot(data = df_viz5, aes(x = top_gilded_subreddit, y = avg_sarcasm, size =avg_gilded_post, color = avg_gilded_unique_subs_count))+
    # setting up x and y axis label, title, and subtitle
    labs(
      x = "10 Most Common User's Top Gilded Subreddits", 
      y = "Sarcasm Rate", 
      size = "Average Gilded\nPost Count", 
      color = "Average Gilded\nUnique Subreddit Count",
      title = "Association between Sarcasm Rate and\nUser's Top Gilded Subreddit",
      subtitle = "To investigate the association between sarcasm on Reddit and the 10 most common user's top gilded subreddits\nalong with user's gilded unique subreddit count and gilded post count."
    ) +
    theme(
      # setting up axis formatting
      text = element_text(size=12, family = 'Helvetica', color = 'white'),
      axis.title = element_text(size=15, family = 'Helvetica', color = 'white'),
      axis.text.x = element_text(size = 12, family = 'Arial', color = 'white', angle = 45, hjust = 1),
      axis.text.y = element_text(size = 12, family = 'Arial', color = 'white'),
      axis.ticks = element_line(colour = "lightgrey"),
      axis.line = element_line(colour = "lightgrey"),
      
      # setting up background formatting
      plot.background = element_rect(fill = "gray60"),
      panel.background = element_rect(fill = "white"),
      panel.grid.major.y  = element_line(color = "gray90", size = 0.5, linetype = 2),
      
      # setting up title and subtitle formatting
      plot.title = element_text(size=20, family = 'Georgia', color = 'white', face='bold'),
      plot.subtitle = element_text(size = 12, family = 'Helvetica', color = 'white', face = 'italic'),
      
      # setting up legend formatting
      legend.text = element_text(size=14, family = 'Helvetica', color = 'white'),
      legend.title = element_text(size=14, family = 'Georgia', color = 'white', face='bold'),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "gray60"),
      legend.key = element_rect(fill = "gray60", colour = NA)
    )
)

arrow_1 = tibble(
  x1 = c('AskReddit'),
  y1 = c(0.4),
  x2 = c('nfl'),
  y2 = c(0.4)
)

# Create plot
plot5 = plot_theme +
  geom_point() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  scale_y_continuous(breaks = seq(0,1,0.1)) +
  ylim(c(0.4, 0.60)) +
  scale_size_continuous(range = c(10, 20)) +
  scale_color_gradient(low = "red", high = "pink", n.breaks = 4) +
  geom_label(x = "funny", y = 0.58, size = 4, color='black', fill = 'grey',
             label = glue::glue("Neutral user's top gilded subreddit like AskReddit and AdviceAnimals tend to be less sarcastic.
                                Average gilded unique subreddit and gilded post count did not seem to have association with sarcasm.")) +
  # input arrow and annotation
  annotate('text', x = "funny", y = 0.41, size = 4, color='black', 
           label = glue::glue("Most Common to Least Common")) + 
  geom_curve(data = arrow_1, 
             aes(x = x1, y = y1, xend = x2, yend = y2),
             arrow = arrow(length = unit(0.07, 'inch')), 
             size = 1.5,
             color = 'darkgrey', 
             curvature = 0)
# Save plot
ggsave(filename = "assets\\visualization\\multivariate_05.png", plot = plot5, width = 10, height = 6, dpi = 300)
