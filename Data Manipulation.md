##Data Manipulation
```{r}
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
```
