'''{r}
###########################################################################################
############################## Kaggle - Titanic Survival ##################################
###########################################################################################

setwd("D:/Study Material/Kaggle/Titanic/Analysis")
train <- read.csv("train.csv",stringsAsFactors=FALSE)
test <- read.csv("test.csv",stringsAsFactors=FALSE)

df.train <- data.frame( Survived=train$Survived,
                     Pclass=train$Pclass,
                     Sex=as.integer(factor(train$Sex)),
                     Age=train$Age,
                     SibSp=train$SibSp,
                     Parch=train$Parch,
                     Fare=train$Fare
                    )
df.test <- data.frame ( Pclass=test$Pclass,
                        Sex=as.integer(factor(test$Sex)),
                        Age=test$Age,
                        SibSp=test$SibSp,
                        Parch=test$Parch,
                        Fare=test$Fare
                    )

median(na.omit(df.test$Age))  ##27
median(na.omit(df.test$Fare)) ##14.4542
df.train$Fare[is.na( df.train$Fare) ] <- 0
df.test$Fare[is.na( df.test$Fare) ]   <- 0
df.train$Age[is.na( df.train$Age) ]     <- 27
df.test$Age[is.na( df.test$Age) ]   <- 27

## Not required for simple models
##train$Survived <- as.factor(train$Survived)


############################################################################################
###################################### Linear Model ########################################
############################################################################################
library("tree")
library("randomForest")

View(df.train)
set.seed(1)  
N = sample(nrow(df.train),round(.8*nrow(df.train)),replace=F)
df1.train = df.train[N,]
df1.valid = df.train[-N,]

## Linear Model ##
set.seed(1)
linear_model <- glm(Survived ~ . , data=df1.train,family=binomial(link="logit"))
summary(linear_model)
Survived_prob <- predict(linear_model, newdata=df1.valid,type="response")
Survived_pred <- ifelse(Survived_prob>0.5,1,0) 
confusion_matrix <- table(Survived_pred,df1.valid$Survived)
TP <- confusion_matrix[2, 2]
TN <- confusion_matrix[1, 1]
FP <- confusion_matrix[2, 1]
FN <- confusion_matrix[1, 2]
recall <- TP / (TP + FN)
specificity <- TN / (FP + TN)
precision <- TP / (TP + FP)
accuracy <- (TP + TN) / nrow(df1.valid)
Misclassification <- 1-accuracy
Misclassification ##0.2078652

library("ROCR")
prediction.lm <- prediction(Survived_prob, df1.valid$Survived)
err.lm <- performance(prediction.lm, measure = "err")
plot(err.lm, ylim=c(0, 0.8))
plot(err.lm, add=T, col="blue")
as.numeric(performance(prediction.lm, "auc")@y.values) 
## Misclassification - 0.2078652
## AUC- 0.8386446

############################################################################################
################################## Ridge Regression ########################################
############################################################################################
library("glmnet")
# test the model.matrix function
m <- model.matrix(Survived ~ . , data= df.train)
head(m)
str(m)

# construct x and y matrix for glmnet()
x <- model.matrix(Survived ~ ., data= df1.train)[,-1]
y <- df1.train$Survived
x.valid <- model.matrix(Survived ~ ., data= df1.valid)[,-1]
y.valid <- df1.valid$Survived

set.seed(1)
ridge.cv <- cv.glmnet(x, y, alpha=0, family="binomial", type.measure="class")
plot(ridge.cv)
# optimal lambda
ridge.lam <- ridge.cv$lambda.min
ridge.lam  ## Lambda= 0.1694972
# plot optimal model

ridge.mod <- glmnet(x, y, alpha=0, family="binomial")
plot(ridge.mod, xvar="lambda", label = TRUE)
abline(v=log(ridge.lam), lty=2)
# prediction
ridge.pred <- predict(ridge.mod, newx=x.valid, s=ridge.lam, type="response", exact=TRUE)
Ridge_pred <- ifelse(ridge.pred>0.5,1,0) 
confusion_matrix_ridge <- table(Ridge_pred,df1.valid$Survived)
TP.ridge <- confusion_matrix_ridge[2, 2]
TN.ridge <- confusion_matrix_ridge[1, 1]
FP.ridge <- confusion_matrix_ridge[2, 1]
FN.ridge <- confusion_matrix_ridge[1, 2]
recall.ridge <- TP.ridge / (TP.ridge + FN.ridge)
specificity.ridge <- TN.ridge / (FP.ridge + TN.ridge)
precision.ridge <- TP.ridge / (TP.ridge + FP.ridge)
accuracy.ridge <- (TP.ridge + TN.ridge) / nrow(df1.valid)
Misclassification.ridge <- 1-accuracy.ridge
Misclassification.ridge ##0.2022472

library("ROCR")
prediction.ridge <- prediction(ridge.pred, y.valid)
err.ridge <- performance(prediction.ridge, measure = "err")
plot(err.ridge, ylim=c(0, 0.8))
plot(err.ridge, add=T, col="blue")
as.numeric(performance(prediction.ridge, "auc")@y.values) ## AUC- 0.8471158

## Misclassification - 0.2022472
## AUC- 0.8471158

############################################################################################
##################################### Random Forest ########################################
############################################################################################
library("tree")
df2.train <- df1.train
df2.train$Survived <- as.factor(df1.train$Survived)
set.seed(1)
titanic.tree <- tree(Survived ~ ., data=df2.train)
summary(titanic.tree)
plot(titanic.tree)
text(titanic.tree) 
## Misclassification rate= 0.1641
## Residual mean deviance= 0.7927

# CV
set.seed(1)
titanic.tree.cv <- cv.tree(titanic.tree)
plot(titanic.tree.cv, type="b")
## Can not prune the tree as best tree is using all variables.

# predict
titanic.tree.prob <- predict(titanic.tree, newdata=df1.valid)[,1]
titanic.tree.pred <- ifelse(titanic.tree.prob>0.5,1,0) 
confusion_matrix_tree <- table(titanic.tree.pred,df1.valid$Survived)
TP.tree <- confusion_matrix_tree[2, 2]
TN.tree <- confusion_matrix_tree[1, 1]
FP.tree <- confusion_matrix_tree[2, 1]
FN.tree <- confusion_matrix_tree[1, 2]
accuracy.tree <- (TP.tree + TN.tree) / nrow(df1.valid)
Misclassification.tree <- 1-accuracy.tree
Misclassification.tree ## 0.8146067 - very high misclassification rate. Lets try random forest

## Random Forests
# fit a random forest model
set.seed(1)
titanic.rf <- randomForest(Survived ~ ., data=df2.train,mtry=2,ntree=501)
set.seed(1)
p <- 6
err.rfs <- rep(0, p)
for(m in 1:p){
  rf <- randomForest(Survived ~ . , data=df2.train, mtry=m, ntree=501)
  cat(m, rf$err.rate[501], "\n")
  err.rfs[m] <- rf$err.rate[501]
}
plot(1:6, err.rfs, type="b", xlab="mtry", ylab="OOB Error")## plotting OOB eror vs m to find optimal m
## mtry=3

set.seed(1)
titanic.rf <- randomForest(Survived ~ ., data=df2.train,mtry=3,ntree=501)
predrfopt = predict(titanic.rf, df1.valid)
confrfopt = table(predrfopt,df1.valid$Survived)
misclassrfopt = (confrfopt[1,2] + confrfopt[2,1])/sum(confrfopt)   #0.1741573

probrf <- predict(titanic.rf, newdata=df1.valid, type="prob")[, 2]
predprobrf <- prediction(probrf, df1.valid$Survived)
ROC.rf <- performance(predprobrf, measure = "tpr", x.measure = "fpr")
plot(ROC.rf)
AUCrf = as.numeric(performance(predprobrf, "auc")@y.values)
AUCrf ##0.8523598
## Misclassification rate- 0.1741573
## AUC - 0.8523598
titanic.rf$importance
varImpPlot(titanic.rf) ## 1. Sex 2. Fare 3. Age
partialPlot(titanic.rf, df1.train, x.var="Age")

##############################
######## SUBMISSION 1 ########
##############################
set.seed(1)
Submit_1 <- test[,1]
Survived <-  predict(titanic.rf, df.test, ntree=501)
Survived_1 <- as.numeric(levels(Survived))[Survived]
Submit_1 <- data.frame(PassengerId=Submit_1,Survived=Survived_1)
#Writing the submission file
cat("saving the submission file\n")
write.csv(Submit_1, "Submit_rf.csv",row.names=FALSE)
## Kaggle Rank- 2128

############################################################################################
#################################### Gradient Boost ########################################
############################################################################################
library(gbm)
require(caret)
# fit a boosting model
df3.train <- df2.train
df3.train$Survived = as.numeric(df2.train$Survived)-1

set.seed(1)
titanic.gbm <- gbm(Survived ~ ., data=df3.train, distribution="bernoulli", 
                  n.trees=5000, interaction.depth=3, shrinkage = .01)
titanic.gbm
str(titanic.gbm)
gbm.perf(titanic.gbm) 
gbm.perf(titanic.gbm, oobag.curve=TRUE)
hist(titanic.gbm$oobag.improve[3000:6000])

# tune gbm by CV
set.seed(1)
titanic.gbm <- gbm(Survived ~ ., data=df3.train, distribution="bernoulli", 
                   n.trees=20000, interaction.depth=3, shrinkage = .001,
                   cv.folds=10) ## runs cross validation
gbm.perf(titanic.gbm) 
plot(titanic.gbm$cv.error, type="l")

# predict
probdgbm <- predict(titanic.gbm, df1.valid, n.trees=8075, type="response")
predgbm = ifelse(probdgbm>0.5,1,0)
confgbm = table(predgbm,df1.valid$Survived)
misclassgbm = (confgbm[1,2] + confgbm[2,1]) / sum(confgbm)   #0.1741573

predprobgbm <- prediction(predgbm, df1.valid$Survived)
ROC.gbm <- performance(predprobgbm, measure = "tpr", x.measure = "fpr")
plot(ROC.gbm)
AUCgbm = as.numeric(performance(predprobgbm, "auc")@y.values)
AUCgbm ##  0.8086594

# TUNING GBM BY CROSS VALIDATION
# ( below code has been commented as it takes a lot of time to run)
n.step <- 20
ds <- c(1, 2, 4, 6, 8)
lambdas <- c(0.01, 0.005, 0.001, 0.0005)
d.size <- length(ds)
l.size <- length(lambdas)
tune.out <- data.frame()

for (i in 1:d.size) {
  for (j in 1:l.size) {
    d <- ds[i]
    lambda <- lambdas[j]
    for (n in (1:10) * n.step / (lambda * sqrt(d))) 
    { set.seed(1)
      titanic.gbm <- gbm(Survived ~ ., data=df3.train, distribution="bernoulli", 
                         n.trees=n, interaction.depth=d, shrinkage=lambda, cv.folds=10)
      n.opt <- gbm.perf(titanic.gbm, method="cv")
      cat("n =", n, " n.opt =", n.opt, "\n")
      if (n.opt / n < 0.95) break
    }
    cv.err <- titanic.gbm$cv.error[n.opt]
    pred.loop <- predict(titanic.gbm, newdata=df1.valid, n.trees=n.opt,type="response")
    pred.loop <- ifelse(pred.loop > 0.5, 1, 0)
    conf.loop <- table(pred.loop, df1.valid$Survived)
    miss.loop <- (conf.loop[1,2]+conf.loop[2,1])/sum(conf.loop)
    out <- data.frame(d=d, lambda=lambda, n=n, n.opt=n.opt, cv.err=cv.err, misclassification=miss.loop)
    print(out)
    tune.out <- rbind(tune.out, out)
  }
}
#After tuning we get the optimal number of trees 
#Optimal values: #trees = 8165, shrinkage = 0.001, depth = 6
# tune gbm by CV
set.seed(1)
titanic.gbm <- gbm(Survived ~ ., data=df3.train, distribution="bernoulli", 
                   n.trees=6948, interaction.depth=6, shrinkage = .001)
gbm.perf(titanic.gbm) 
probdgbm <- predict(titanic.gbm, df1.valid, n.trees=6948, type="response")
predgbm = ifelse(probdgbm>0.5,1,0)
confgbm = table(predgbm,df1.valid$Survived)
misclassgbm = (confgbm[1,2] + confgbm[2,1]) / sum(confgbm)#0.1516854
misclassgbm
predprobgbm <- prediction(predgbm, df1.valid$Survived)
ROC.gbm <- performance(predprobgbm, measure = "tpr", x.measure = "fpr")
plot(ROC.gbm)
AUCgbm = as.numeric(performance(predprobgbm, "auc")@y.values)
AUCgbm ##0.8251311

summary(titanic.gbm)
plot(titanic.gbm)
plot(titanic.gbm, i=c("Sex", "Fare"))
## So looks like Random Forest is giving better result. Now to improve the result lets try XGBoost

# prob_ensemble <- (0.8 * probrf + 0.2 * probdgbm) 
# pred_ensemble = ifelse(prob_ensemble>0.5,1,0)
# conf_ensemble = table(pred_ensemble,df1.valid$Survived)
# misclass_ensemble = (conf_ensemble[1,2] + conf_ensemble[2,1]) / sum(conf_ensemble)#0.1516854
# misclass_ensemble
# predprob_ensemble <- prediction(pred_ensemble, df1.valid$Survived)
# ROC.ensemble <- performance(predprob_ensemble, measure = "tpr", x.measure = "fpr")
# plot(ROC.ensemble)
# AUC_ensemble = as.numeric(performance(predprob_ensemble, "auc")@y.values)
# AUC_ensemble ## 0.7966922



############################################################################################
########################################## XG Boost ########################################
############################################################################################
# XGboost with random values of parameters; like GBM, XGboost also requires matrix
require(xgboost)
train.mx1 = sparse.model.matrix (Survived ~., df3.train)
valid.mx1 = sparse.model.matrix (Survived ~. , df1.valid)
dtrain1 = xgb.DMatrix(train.mx1, label=df3.train$Survived)
dvalid1 = xgb.DMatrix(valid.mx1,  label=df1.valid$Survived)

set.seed(1)
## Create a Grid; 3*3*5=45 combinations
xgb_grid = expand.grid( nrounds = c(5000,10000),
                          eta = c(0.001,0.004,0.008),
                          max_depth = c(3,4,5))

## Pack the training control parameters
xgb_trcontrol = trainControl( method = "cv",number = 4)
# Train the model for each parameter combination in the grid, using CV to evaluate
xgb_train = train(Survived ~.  ,trControl = xgb_trcontrol,
                                  tuneGrid = xgb_grid,
                                  method = "xgbTree",
                                  data=df2.train,   ## factors required as inputs
                                  metric='Accuracy')
xgb_train$bestTune
xgb_train$results
# nrounds=2000,  max_depth=4,  eta=0.001
#Using the optimal value of the parameters on the modified dataset
set.seed(1)
titanic.xgb = xgb.train(params=list(objective='reg:logistic',eta=0.0001,max_depth=4),
                        data=dtrain1, 
                        nrounds=2000, 
                        watchlist=list(eval=dvalid1, train=dtrain1), 
                        maximize=FALSE)

probxgb <- predict(titanic.xgb,newdata= dvalid1)
predxgb = ifelse(probxgb>0.5,1,0)
confxgb = table(predxgb,df1.valid$Survived)
misclassxgb = (confxgb[1,2] + confxgb[2,1]) / sum(confxgb)#0.1292135
misclassxgb
predprobxgb <- prediction(predxgb, df1.valid$Survived)
ROC.xgb <- performance(predprobxgb, measure = "tpr", x.measure = "fpr")
plot(ROC.xgb)
AUCxgb = as.numeric(performance(predprobxgb, "auc")@y.values)
AUCxgb ##0.8520237

set.seed(1)
Submit_2 <- test[,1]
Survived_2 <-  predict(titanic.xgb, as.matrix(df.test))
Survived_2 <- ifelse(Survived_2>0.5,1,0)
Submit_2 <- data.frame(PassengerId=Submit_2,Survived=Survived_2)
cat("saving the submission file\n")
write.csv(Submit_2, "Submit_xgb.csv",row.names=FALSE)


############################################################################################
############################ XG Boost- Different tuning ####################################
############################################################################################
params <- expand.grid(max_depth = c(2,4,6,8), eta = c(0.1,0.001,0.004,0.008), 
                      min_child_weight = c(1,10), subsample = c(1,0.5))
for (k in 1:nrow(params)) 
{
  prm <- params[k,]
  print(prm)
  print(system.time({
    n_proc <- detectCores()
    md <- xgb.train(data = dtrain1, nthread = n_proc, 
                    objective = "binary:logistic", nround = 5000, 
                    max_depth = prm$max_depth, eta = prm$eta, 
                    min_child_weight = prm$min_child_weight, subsample = prm$subsample, 
                    watchlist = list(valid = dvalid1, train = dtrain1), eval_metric = "auc",
                    early_stop_round = 100, printEveryN = 100)
  }))
}

probxgb2 <- predict(md, newdata = dvalid1)
predprobxgb2 <- prediction(probxgb2, df1.valid$Survived)
ROC.xgb2 <- performance(predprobxgb2, measure = "tpr", x.measure = "fpr")
plot(ROC.xgb2)
AUCxgb2 = as.numeric(performance(predprobxgb2, "auc")@y.values)
AUCxgb2 ##0.8602259

set.seed(1)
Submit_3 <- test[,1]
Survived_3 <-  predict(md, as.matrix(df.test))
Survived_3 <- ifelse(Survived_3>0.5,1,0)
Submit_3 <- data.frame(PassengerId=Submit_3,Survived=Survived_3)
cat("saving the submission file\n")
write.csv(Submit_3, "Submit_xgb_new.csv",row.names=FALSE)

############################################################################################
###################################### Train using RF ######################################11
############################################################################################

setwd("D:/Study Material/Kaggle/Titanic/Analysis")
train <- read.csv("train.csv",stringsAsFactors=FALSE, header = TRUE)
test <- read.csv("test.csv",stringsAsFactors=FALSE, header = TRUE)
train$Survived <- factor(train$Survived)
median(na.omit(train$Age))  ##28
median(na.omit(test$Fare)) ##14.4542
train$Fare[is.na( train$Fare) ] <- 0
test$Fare[is.na( test$Fare) ]   <- 0
train$Age[is.na( train$Age) ] <- 28
test$Age[is.na( test$Age) ]   <- 28

# Set a random seed (so you will get the same results as me)
set.seed(1)
# Train the model using a "random forest" algorithm
model <- train(Survived ~ Pclass + Sex + SibSp + Age+  
                 Embarked + Parch + Fare, # Survived is a function of the variables we decided to include
               data = train, # Use the trainSet dataframe as the training data
               method = "rf",# Use the "random forest" algorithm
               trControl = trainControl(method = "cv", # Use cross-validation
                                        number = 5) # Use 5 folds for cross-validation
)

model

test$Survived  <- predict(model,newdata= test)
summary(test)
submission <- test[,c("PassengerId", "Survived")]
write.table(submission, file = "Submit_Embarked.csv", col.names = TRUE, row.names = FALSE, sep = ",")


############################################################################################
############################### Feature Engineering-title ##################################
############################################################################################

setwd("D:/Study Material/Kaggle/Titanic/Analysis")
train <- read.csv("train.csv",stringsAsFactors=FALSE, header = TRUE)
test <- read.csv("test.csv",stringsAsFactors=FALSE, header = TRUE)
train$Survived <- factor(train$Survived)
median(na.omit(train$Age))  ##28
median(na.omit(test$Fare)) ##14.4542
train$Fare[is.na( train$Fare) ] <- 0
test$Fare[is.na( test$Fare) ]   <- 0
train$Age[is.na( train$Age) ] <- 28
test$Age[is.na( test$Age) ]   <- 28

# getTitle <- function(data) {
#   title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
#   title.comma.end <- title.dot.start 
#   + attr(title.dot.start, "match.length")-1
#   data$Title <- substr(data$Name, title.dot.start+2, title.comma.end-1)
#   return (data$Title)
# }   

train$title <- str_sub(train$Name, str_locate(train$Name, ",")[ , 1] + 2, str_locate(train$Name, "\\.")[ , 1] - 1)
male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
train$title[train$title %in% male_noble_names] <- "male_noble"
female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess","Dona")
train$title[train$title %in% female_noble_names] <- "female_noble"

test$title <- str_sub(test$Name, str_locate(test$Name, ",")[ , 1] + 2, str_locate(test$Name, "\\.")[ , 1] - 1)
male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
test$title[test$title %in% male_noble_names] <- "male_noble"
female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess","Dona")
test$title[test$title %in% female_noble_names] <- "female_noble"

# Set a random seed (so you will get the same results as me)
set.seed(1)
# Train the model using a "random forest" algorithm
model <- train(Survived ~ Pclass + Sex + SibSp + Age+  title+
                 Embarked + Parch + Fare, # Survived is a function of the variables we decided to include
               data = train, # Use the trainSet dataframe as the training data
               method = "rf",# Use the "random forest" algorithm
               trControl = trainControl(method = "cv", # Use cross-validation
                                        number = 5) # Use 5 folds for cross-validation
)
model

test$Survived  <- predict(model,newdata= test)
summary(test)
submission <- test[,c("PassengerId", "Survived")]
write.table(submission, file = "Submit_title.csv", col.names = TRUE, row.names = FALSE, sep = ",")
'''
