## Linear Model, Ridge Regression and Random Forest

```{r}
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

############################################################################################
set.seed(1)
Submit_1 <- test[,1]
Survived <-  predict(titanic.rf, df.test, ntree=501)
Survived_1 <- as.numeric(levels(Survived))[Survived]
Submit_1 <- data.frame(PassengerId=Submit_1,Survived=Survived_1)
#Writing the submission file
cat("saving the submission file\n")
write.csv(Submit_1, "Submit_rf.csv",row.names=FALSE)
```
## Kaggle Rank- 2128
