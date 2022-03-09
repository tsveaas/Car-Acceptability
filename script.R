##########################################
# Capstone Project - Choose your own
##########################################

# RMarkdown settings
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(fig.width=5, fig.height=2.5, fig.align="center") 
knitr::opts_chunk$set(fig.pos = "H", out.extra = "")

# required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(xfun)) install.packages("xfun", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(knitr)
library(magrittr)
library(xfun)
library(tinytex)

#Load required packages used in project
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(stringr)
library(rpart.plot)
library(randomForest)

# Downloading the car data set
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", dl)

car_data <- str_split_fixed(readLines(dl), ",", 7)
colnames(car_data) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety","class")
car_data <- as.data.frame(car_data)

# Change the class section to just acc and unacc to simplify set
car_data <- car_data %>%
  mutate(class = if_else(class == "unacc", "unacc" , "acc"))


# Validation set will be 20% of the cars data 
set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = car_data$class, times = 1, p = 0.2, list = FALSE)
edx <- car_data[-validation_index,]
validation <- car_data[validation_index,]

# Explore the edx data set
dim(edx)
structure(edx)
unique(edx$class)
unique(edx$buying)
unique(edx$maint)
unique(edx$doors)
unique(edx$persons)
unique(edx$lug_boot)
unique(edx$safety)

# Analyze and Visualize the data and relation to class

# See ratio of class'
edx %>% ggplot(aes(x = class)) +
  geom_histogram(stat = "count") +
  ggtitle("Car Acceptability Class") +
  xlab("Class") +
  ylab("Count") +
  theme_stata()

# Compare buying price and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(buying = factor(buying, levels = c("low", "med", "high", "vhigh"))) %>%
  ggplot(aes(fill = class, x = buying)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Buying Price and Class") +
  xlab("Buying Price") +
  ylab("Count")

# Comparing Maintenance Costs and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(maint = factor(maint, levels = c("low", "med", "high", "vhigh"))) %>%
  ggplot(aes(fill = class, x = maint)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Maintenance Costs and Class") +
  xlab("Maintenance Costs") +
  ylab("Count")

# Comparing Doors and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(doors = factor(doors, levels = c("2", "3", "4", "5more"))) %>%
  ggplot(aes(fill = class, x =doors)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Doors and Class") +
  xlab("Doors") +
  ylab("Count")

# Comparing person capacity and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(persons = factor(persons, levels = c("2", "4", "more"))) %>%
  ggplot(aes(fill = class, x = persons)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Person Capacity and Class") +
  xlab("Person Capacity") +
  ylab("Count")

# Comparing Size of boot and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(lug_boot = factor(lug_boot, levels = c("small", "med", "big"))) %>%
  ggplot(aes(fill = class, x = lug_boot)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Size of Luggage Boot and Class") +
  xlab("Size of Luggage Boot") +
  ylab("Count")

# Comparing Safety Level and Class
edx %>% mutate(class = factor(class, levels = c("unacc", "acc"))) %>%
  mutate(safety = factor(safety, levels = c("low", "med", "high"))) %>%
  ggplot(aes(fill = class, x = safety)) +
  geom_histogram(position = "stack", stat = "count") +
  ggtitle("Safety Level and Class") +
  xlab("Safety Level") +
  ylab("Count")

################################
# Creating the Model
################################

# Further split the data into a train and test set
# Test set will be 20% of the edx data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$class, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Building the decision tree model
# Do repeated cross validation with 10 folds 3 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Create the decision tree using train function
set.seed(1, sample.kind="Rounding")
decision_tree <- train(class ~ ., data = train_set, method = "rpart",
                       trControl = train_control,
                       tuneLength = 10)

# Plot the decision tree model
prp(decision_tree$finalModel, box.palette = "auto",
    faclen = 0, # print fullt factor names
    varlen = 0) # write full names in branches

# Use the model to predict test set
test_pred_rpart <- predict(decision_tree, test_set)
confusionMatrix(test_pred_rpart, as.factor(test_set$class))


# Build random forest model
tunegrid <- expand.grid(mtry = c(1:5), ntrees = c(50,100,150,200))
set.seed(1, sample.kind="Rounding")
random_forest <- train(class ~ ., data = train_set, method = "rf",
                              metric = "Accuracy",
                              tunegrid = tunegrid,
                              trControl = train_control)

# USe random forest model to predict test set
test_pred_rf <- predict(random_forest, test_set)
confusionMatrix(test_pred_rf, as.factor(test_set$class))

# Use decision tree model on the validation set
valid_pred <- predict(decision_tree, validation)
confusionMatrix(valid_pred, as.factor(validation$class))