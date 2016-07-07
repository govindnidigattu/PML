# PML
The objective and goal of this project is to predict the manner in which they performed the exercise and machine learning classification of accelerometers data on the belt, forearm, arm, and dumbell of 6 participants.In training data "classe" is the outcome variable in the training set using predictor variables to predict 20 different test cases
---
title: "Practical Machine Learning-Prediction Assignment Writeup"
author: "Dr.Govind Nidigattu"
date: "June 28, 2016"
output: html_document
---

The objective and goal of this project is to predict the manner in which they performed the exercise and machine learning classification of accelerometers data on the belt, forearm, arm, and dumbell of 6 participants.In training data "classe" is the outcome variable in the training set using predictor variables to predict 20 different test cases.The data for this project come from this source is: http://groupware.les.inf.puc-rio.br/har.

The "classe" variable which classifies the correct and incorrect outcomes of A, B, C, D, and E categories. Coursera project writeup describes the model cross validation and expected out of sample error rate. Models applied successfully to predict all 20 different test cases on the Coursera website.

This project work partially submitted to the Coursera for the course "Practical Machine Learning" by Jeff Leek, PhD, Professor at Johns Hopkins University, Bloomberg School of Public Health.

```{r}
# The training and testing datasets for this project are available here below web resources
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
```

```{r}
library(AppliedPredictiveModeling)
library(rpart)
library(ElemStatLearn)
library(pgmm)
library(caret)
library(e1071)
library(gbm) 
library(lubridate)
library(plyr)
library(elasticnet)
library(C50)
library(car)
```


```{r}
## Removing NA values from training data
training1 <- read.csv("D://coursera//Practical Machine Learning//training.csv", header=T)
noNAtraining <- training1[ , colSums(is.na(training1)) == 0]
dim(noNAtraining)
```

```{r}
## Removing Variables with user related details by step by step
removetraining<-noNAtraining[,-c(1:7)]
removetraining1<-removetraining[,-c(5:13)]
removetraining2<-removetraining1[,-c(27:32)]
removetraining3<-removetraining2[,-c(30:39)]
removetraining4<-removetraining3[,-c(42:50)]
dim(removetraining4)
```

```{r}
# Removing highly correlated variables from training data
removetraining4.nonzerovar <- removetraining4
corrMatrix <- cor(na.omit(removetraining4.nonzerovar[sapply(removetraining4.nonzerovar, is.numeric)]))
dim(corrMatrix)
```

Preprocessing of NA values and user related information has been done for both the testing and training datasets.Removed the variables that have extremely low variance and highly correlated.

```{r}
training <- read.csv("D://coursera//Practical Machine Learning//pmlmodeling//training.csv", header=T)
testing <- read.csv("D://coursera//Practical Machine Learning//pmlmodeling//testing.csv", header=T)
```

```{r}
#Data Partitioning and Prediction Process
library(caret)
trainingfinal <- training
inTrain<-createDataPartition(y=trainingfinal$classe, p=0.7,list = FALSE)
train<-trainingfinal[inTrain,] 
test<-trainingfinal[-inTrain,] 
```

```{r}
# Relative weights function
relweights <- function(fit,...){                           
  R <- cor(fit$model)     
  nvar <- ncol(R)            
  rxx <- R[2:nvar, 2:nvar]   
  rxy <- R[2:nvar, 1]
  svd <- eigen(rxx)          
  evec <- svd$vectors                             
  ev <- svd$values           
  delta <- diag(sqrt(ev))    
  lambda <- evec %*% delta %*% t(evec)          
  lambdasq <- lambda ^ 2     
  beta <- solve(lambda) %*% rxy             
  rsquare <- colSums(beta ^ 2)                     
  rawwgt <- lambdasq %*% beta ^ 2      
  import <- (rawwgt / rsquare) * 100   
  lbls <- names(fit$model[2:nvar])     
  rownames(import) <- lbls  
  colnames(import) <- "Weights"  
  barplot(t(import),names.arg=lbls,
          ylab="% of R-Square",
          xlab="Predictor Variables",
          main="Relative Importance of Predictor Variables",      
          sub=paste("R-Square=", round(rsquare, digits=3)),    
          ...) 
  return(import) 
  }
```

```{r}
training <- read.csv("D://coursera//Practical Machine Learning//pmlmodeling//training.csv", header=T)
dtrain <- training
dtrain <- data.frame(dtrain)
dtrain <- data.frame(sapply(dtrain,as.numeric))
fit <- lm(classe ~ ., data=dtrain)
relweights(fit,col="blue")

```

Ralative importance of predictor variables based on weights are magnet_belt_y(11.3181850), magnet_dumbbell_z(12.2589816),pitch_forearm(11.2184124), pitch_forearm(11.2184124), accel_forearm_z(6.2238297), total_accel_forearm  (4.7259294),accel_arm_z(3.3813379), magnet_arm_x(3.2271041) and magnet_arm_y(3.5409904). 

```{r}
set.seed(3141592)
fitModel <- train(classe~.,data=train,method="rf",trControl=trainControl(method="cv",number=10),
                  prox=TRUE,
                  verbose=TRUE,
                  allowParallel=TRUE)
predictions <- predict(fitModel, newdata=test)
confusionMat <- confusionMatrix(predictions, test$classe)
confusionMat
```

In modeling using a Random Forest algorithm with 44 variables which are among the significant variables.Applied a 10-fold cross-validation. Model accuracy is 99.9% with 44 predictor variables.

```{r}
#Training and test set dimensions
dim(train)
dim(test)
```

```{r}
# Classification by c50 algorithm
library(C50)
C50.model0 <- C5.0(classe ~ ., data = training,rules=TRUE)
test.C50.model0 <- predict.C5.0(C50.model0, testing, type = "class")
test.C50.model0.df <- as.data.frame(test.C50.model0)
names(test.C50.model0.df)
test.C50.model0.df$id <- testing$id
print(test.C50.model0.df)
```

```{r}
# Classification by random forest
library(randomForest)
set.seed(1234)
fit.forest <- randomForest(classe~., data=training,        
                           na.action=na.roughfix,
                           importance=TRUE)             
fit.forest
varImpPlot(fit.forest)
importance(fit.forest, type=2)                          
forest.pred <- predict(fit.forest, newdata=testing)
forest.pred
```

```{r}
# Classification by SVM
library(e1071)
set.seed(1234)
df.train <- training
df.validate <- testing
table(df.train$classe)
table(df.validate$classe)
fit.svm <- svm(classe~., data=df.train)
fit.svm
svm.pred <- predict(fit.svm, na.omit(df.validate))
svm.pred
```
