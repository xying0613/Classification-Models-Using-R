setwd("C:/Users/User/Documents/FIT3152/Assignment 02")

library(tree)
library(e1071)
library(adabag)
library(randomForest)
library(ROCR)
library(caret)

rm(list = ls())

WAUS <- read.csv("CloudPredict2021.csv")
L <- as.data.frame(c(1:49))

set.seed(29797918)

L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows


# ------------------------------- QUESTION 1 & 2 ------------------------------- #

# Look at structure of WAUS
str(WAUS)

# Look at the summary of each independent variables before making any changes
summary(WAUS[,-23])

# Remove rows that are N/A in CloudTomorrow and modify class of variables below
WAUS <- WAUS[-which(is.na(WAUS$CloudTomorrow)),]
WAUS$Day <- as.factor(WAUS$Day)
WAUS$Month <- as.factor(WAUS$Month)
WAUS$Year <- as.factor(WAUS$Year)
WAUS$WindGustDir <- as.factor(WAUS$WindGustDir)
WAUS$WindDir9am <- as.factor(WAUS$WindDir9am)
WAUS$WindDir3pm <- as.factor(WAUS$WindDir3pm)
WAUS$RainToday <- as.factor(WAUS$RainToday)
WAUS$CloudTomorrow <- as.factor(WAUS$CloudTomorrow)

# Find proportion of cloudy to not cloudy days
sum(WAUS$CloudTomorrow==0)
sum(WAUS$CloudTomorrow==1)
sum(WAUS$CloudTomorrow==1)/sum(WAUS$CloudTomorrow==0)

# Look at the summary of current independent variables
summary(WAUS[,-23])

# Standard deviation and variation of each independent variables (only numeric attriutes)
lapply(WAUS[complete.cases(WAUS[,c(5:9, 11, 14:21)]), c(5:9, 11, 14:21)], sd)
lapply(WAUS[complete.cases(WAUS[,c(5:9, 11, 14:21)]), c(5:9, 11, 14:21)], var)

# Function to return mode of the attribute (column)
getMode <- function(column) {
  freq <- as.data.frame(table(column))
  freq <- freq[order(freq$Freq, decreasing = TRUE),]
  return(as.factor(freq[1,1]))
}

# If numeric, fill in N/As with mean; if factor, fill in N/As with mode
for (i in 1:22){
  if (class(WAUS[,i]) == "factor"){
    WAUS[is.na(WAUS[,i]), i] <- getMode(WAUS[,i])
  } else {
    WAUS[is.na(WAUS[,i]), i] <- mean(WAUS[,i], na.rm=TRUE)
  }
}

# Look at the summary of independent variables after replacing N/A
summary(WAUS[,-23])


# --------------------------------- QUESTION 3 --------------------------------- #

# Split data to 70% training set and 30% test set
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
WAUS.train = WAUS[train.row,]
WAUS.test = WAUS[-train.row,]


# --------------------------------- QUESTION 4 --------------------------------- #

# Fitting the DECISION TREE model
tree.fit <- tree(CloudTomorrow~., data = WAUS.train)

# Fitting the NAIVE BAYES model 
nb.fit <- naiveBayes(CloudTomorrow~., data = WAUS.train)

# Fitting the BAGGING model
bag.fit <- bagging(CloudTomorrow~., data = WAUS.train)

# Fitting the BOOSTING model
boost.fit <- boosting(CloudTomorrow~., data = WAUS.train)

# Fitting the RANDOM FOREST model
rf.fit <- randomForest(CloudTomorrow~., data = WAUS.train)


# --------------------------------- QUESTION 5 --------------------------------- #

# Making predictions for DECISION TREE from test data
tree.predict <- predict(tree.fit, WAUS.test, type="class")

# Making predictions for NAIVE BAYES from test data
nb.predict <- predict(nb.fit, WAUS.test)

# Making predictions for BAGGING from test data
bag.predict <- predict.bagging(bag.fit, WAUS.test)

# Making predictions for BOOSTING from test data
boost.predict <- predict.boosting(boost.fit, WAUS.test)

# Making predictions for RANDOM FOREST from test data
rf.predict <- predict(rf.fit, WAUS.test)

# Confusion Matrix of Decision Tree, Naive Bayes, Bagging, Boosting and Random Forest accordingly
confusionMatrix(data = tree.predict, reference = as.factor(WAUS.test$CloudTomorrow))
confusionMatrix(data = nb.predict, reference = as.factor(WAUS.test$CloudTomorrow))
confusionMatrix(data = as.factor(bag.predict$class), reference = as.factor(WAUS.test$CloudTomorrow))
confusionMatrix(data = as.factor(boost.predict$class), reference = as.factor(WAUS.test$CloudTomorrow))
confusionMatrix(data = rf.predict, reference = as.factor(WAUS.test$CloudTomorrow))


# --------------------------------- QUESTION 6 --------------------------------- #

# ROC curve for DECISION TREE
t.pred <- prediction(predict(tree.fit, WAUS.test, type="vector")[,2], WAUS.test$CloudTomorrow)
t.perf <- performance(t.pred,"tpr","fpr")
plot(t.perf, col=3, lwd=2, main="ROC curves of different machine learning classifier")

# ROC curve for NAIVE BAYES
nb.pred <- prediction(predict(nb.fit, WAUS.test, type="raw")[,2], WAUS.test$CloudTomorrow)
nb.perf <- performance(nb.pred,"tpr","fpr")
plot(nb.perf, col=4, lwd=2, add=TRUE)

# ROC curve for BAGGING
bag.pred <- prediction(predict(bag.fit, WAUS.test)$prob[,2], WAUS.test$CloudTomorrow)
bag.perf <- performance(bag.pred,"tpr","fpr")
plot(bag.perf, col=5, lwd=2, add=TRUE)

# ROC curve for BOOSTING
boost.pred <- prediction(predict(boost.fit, WAUS.test)$prob[,2], WAUS.test$CloudTomorrow)
boost.perf <- performance(boost.pred,"tpr","fpr")
plot(boost.perf, col=6, lwd=2, add=TRUE)

# ROC curve for RANDOM FOREST
rf.pred <- prediction(predict(rf.fit, WAUS.test, type="prob")[,2], WAUS.test$CloudTomorrow)
rf.perf <- performance(rf.pred,"tpr","fpr")
plot(rf.perf, col=7, lwd=2, add=TRUE)

legend(0.6, 0.4, c("Decision Tree","Naive Bayes","Bagging","Boosting","Random Forest"), 3:7)

# AUC
cat("Area under curve for Decision Tree:", as.numeric(performance(t.pred, "auc")@y.values))
cat("Area under curve for Naive Bayes:", as.numeric(performance(nb.pred, "auc")@y.values))
cat("Area under curve for Bagging:", as.numeric(performance(bag.pred, "auc")@y.values))
cat("Area under curve for Boosting:", as.numeric(performance(boost.pred, "auc")@y.values))
cat("Area under curve for Random Forest:", as.numeric(performance(rf.pred, "auc")@y.values))


# --------------------------------- QUESTION 8 --------------------------------- #

# Importance of variables for each model
summary(tree.fit)
bag.fit$importance
boost.fit$importance
rf.fit$importance


# --------------------------------- QUESTION 9 --------------------------------- #

# Do cross validation to find out the best tree size (least error)
cv <- cv.tree(tree.fit, FUN=prune.misclass, K=20)
plot(cv$size, cv$dev, type="b", xlab="Tree Size", ylab="Error Rate", main="Cross Validation: Error Vs Size")

# View the size and error
cv$size
cv$dev

# Prune the tree according to results above
new.tree <- prune.misclass(tree.fit, best=6)

# Predict this simple tree
new.tree.predict <- predict(new.tree, WAUS.test, type="class")

# Look at the summary of this pruned simple tree
summary(new.tree)

# Plot the simple tree
plot(new.tree)
text(new.tree, pretty = 12)

# Confusion matrix and statistics of this tree
confusionMatrix(data = new.tree.predict, reference = as.factor(WAUS.test$CloudTomorrow))

# AUC of this pruned simple tree
new.pred <- prediction(predict(new.tree, WAUS.test, type="vector")[,2], WAUS.test$CloudTomorrow)
cat("Area under curve for pruned Decision Tree:", as.numeric(performance(new.pred, "auc")@y.values))


# -------------------------------- QUESTION 10 --------------------------------- #

# Do 5-folds cross validation (2 times) and hyperparameter tuning (mtry)
control <- trainControl(method='repeatedcv', number=5, repeats=2, search='grid')
tunegrid <- expand.grid(.mtry=c(1:22))
rft.fit <- train(CloudTomorrow ~., data = WAUS.train, method = 'rf',metric = 'Accuracy', tuneGrid = tunegrid, trControl=control)

# Look at the results
print(rft.fit)

# Make predictions using test data
rft.predict <- predict(rft.fit, WAUS.test)

# Confusion Matrix and statistics of this model
confusionMatrix(data = rft.predict, reference = as.factor(WAUS.test$CloudTomorrow))

# AUC
rft.pred <- prediction(predict(rft.fit, WAUS.test, type="prob")[,2], WAUS.test$CloudTomorrow)
rft.perf <- performance(rft.pred,"tpr","fpr")
cat("Area under curve for modified Random Forest:", as.numeric(performance(rft.pred, "auc")@y.values))

# Plot the ROC curves for all classifiers until now to compare
plot(t.perf, col=3, lwd=2, main="ROC curves of different machine learning classifier")
plot(nb.perf, col=4, lwd=2, add=TRUE)
plot(bag.perf, col=5, lwd=2, add=TRUE)
plot(boost.perf, col=6, lwd=2, add=TRUE)
plot(rf.perf, col=7, lwd=2, add=TRUE)
plot(rft.perf, col=9, lwd=2, add=TRUE)

legend(0.55, 0.4, c("Decision Tree","Naive Bayes","Bagging","Boosting","Random Forest","Modified Random Forest"), c(3:7,9))


# -------------------------------- QUESTION 11 --------------------------------- #

library(neuralnet)

# Make a copy for training set and testing set
neural.train <- WAUS.train
neural.test <- WAUS.test

# Convert wind directions to integer, change factors to numeric (Pre-processing)
dir <- setNames(seq(0, 337.5 , by=22.5), c("N","NNE","NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"))
neural.train$Day <- as.numeric(neural.train$Day)
neural.train$Month <- as.numeric(neural.train$Month)
neural.train$Year <- as.numeric(neural.train$Year)
neural.train$WindGustDir <- dir[neural.train$WindGustDir]
neural.train$WindDir9am <- dir[neural.train$WindDir9am]
neural.train$WindDir3pm <- dir[neural.train$WindDir3pm]
neural.train$RainToday <- ifelse(neural.train$RainToday=="Yes", 1, 2)
neural.train$CloudTomorrow <- ifelse(neural.train$CloudTomorrow==0, 0, 1)

neural.test$Day <- as.numeric(neural.test$Day)
neural.test$Month <- as.numeric(neural.test$Month)
neural.test$Year <- as.numeric(neural.test$Year)
neural.test$WindGustDir <- dir[neural.test$WindGustDir]
neural.test$WindDir9am <- dir[neural.test$WindDir9am]
neural.test$WindDir3pm <- dir[neural.test$WindDir3pm]
neural.test$RainToday <- ifelse(neural.test$RainToday=="Yes", 1, 2)
neural.test$CloudTomorrow <- ifelse(neural.test$CloudTomorrow==0, 0, 1)

# Normalize data (min-max scaling)
normalization <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (i in 1:22){
  neural.train[,i] <- normalization(neural.train[,i])
  neural.test[,i] <- normalization(neural.test[,i])
}

# Fit the ANN model
ann <- neuralnet(CloudTomorrow~., data=neural.train, hidden=3, threshold=0.01)

# Plot ANN model
plot(ann)

# Look at the result matrix of ANN model
ann$result.matrix

# Make prediction using test data
prediction <- compute(ann, neural.test)
prob <- prediction$net.result
pred <- ifelse(prob > 0.5, 1, 0)

# Confusion Matrix of ANN model
confusionMatrix(data = as.factor(pred), reference = as.factor(WAUS.test$CloudTomorrow))







x=c(1,4)
y=c("hi","bye")
z=c(a)
class(z)
a=as.factor(y)


max.type <- by(iris, iris[5], function(df)
  df[which.max(df[,3]),])

do.call(rbind, max.type)
