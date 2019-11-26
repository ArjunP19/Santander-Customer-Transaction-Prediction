#Clean the environment 
library(DataCombine)
library(ggplot2)
library(gridExtra)
library(caret)
library(randomForest)
library(ISLR)
library(glmnet)
library(ROCR)
library(pROC)
library(e1071)

#remove all the objects storage.mode
rm(list=ls())



# choose the Train dataset
print("Choose the TRAIN DATASET")
train_file <- file.choose()

#choose the Test dataset
print("Choose the TEST DATASET")
test_file <- file.choose()

# Load the cvs files Train and Test
santander_df1 = read.csv(train_file, header = F)
head(santander_df1)

test_df<-read.csv(test_file)
head(test_df)



#view columns
colnames(santander_df)

#view columns
colnames(test_df)

#view the structure of Train dataframe
str(santander_df)

#view the structure of Test dataframe
str(test_df)

#view diminesion of Train dataframe
dim(santander_df)

#view dimension of Test dataframe
dim(test_df)



#Summary to view dataset stats 
summary(santander_df)

summary(test_df)


class(santander_df$target)

#convert traget class to factor
santander_df$target<-as.factor(santander_df$target)
class(santander_df$target)

require(gridExtra)
#Count of target classes 0 and 1
table(santander_df$target)
table(santander_df$target)

#Percenatge counts of target classes (0 and 1)
table(santander_df$target)/length(santander_df$target)*100

#Bar plot for count of target classes
ggplot(santander_df,aes(target))+theme_bw()+geom_bar(stat='Count',fill='lightblue')+xlab("Target Class")+
  ylab("Total Count")+ggtitle("Total count of Class [0 and 1]")+theme(text = element_text(size = 13))



####Distribution of train attributes from 3 to 80
for (var in names(santander_df)[c(3:20)]){
  target<-santander_df$target
 plot3<-ggplot(santander_df, aes(x=santander_df[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot3)
}




#Distribution of test attributes from 2 to 80
#ggplot(test_df[,c(2:80)], ggtheme = theme_classic(),geom_density_args = list(color='cyan'))



#Finding the missing values in train and test data, on whole table 
#table(is.na(santander_df))
#table(is.na(test_df))


#Finding the missing values in train data, using apply function on column computation 
missing_value<-data.frame(missing_value=apply(santander_df,1,function(x){sum(is.na(x))}))
missing_value<-sum(missing_value)
missing_value



#Finding the missing values in test data, using apply function on column computation 
missing_value<-data.frame(missing_value=apply(test_df,2,function(x){sum(is.na(x))}))
missing_value<-sum(missing_value)
missing_value


###
####  METHOD 1 - Random Forest
###

#Divide the data into Train and Test using stratified sample method for 

train.index = createDataPartition(santander_df$target, p = .75, list = FALSE)
train = santander_df[train.index,]
test = santander_df[-train.index,]

dim(train)
dim(test)



str(santander_df)
#setting the mtry
mtry<-floor(sqrt(23))

###
####Random Forest classifier to develop model on training data
###


RF_model = randomForest(target~., train[, -c(1)], mtry=mtry, importance=TRUE, ntree= 100)



#Predicit the test data using Random Forest model
RF_Predicition = predict(RF_model, test[, -c(1)])
RF_Predicition


#Evaluate the performance of the classification model
COnf_matrix = table(RF_Predicition, test$target)
confusionMatrix(COnf_matrix)


X_test = data.matrix(test[, -c(1,2)])
y_test_traget = data.matrix(test$target)

pred1<-as.integer(RF_Predicition)
Test1 <-as.factor(X_test)
Target1<-as.factor(y_test_traget)

## ROC Curve
roc_score=roc(data=Test1, response=Target1, predictor=pred1, auc=TRUE, plot=TRUE)
plot(roc_score)
print(roc_score)

##Predict the Model with TEST DATASET
TEST_DF<-as.matrix(test_df[,-c(1)])

colnames(TEST_DF)
## Predict the Model on TEST dataset
TestRF_predict <- predict(RF_model, TEST_DF, type = 'class')
print(TestRF_predict)


######
### First tried with Random Forest model with stratified sampling method, we can observer that Accuracy : 0.8995 and 
### AUC curve 0.5, Model is predicting the 89 percent of accuracy but looking at the ROC/AUC curve and other vaules
### model is not good on unbalanced data and it's taking more time in training and predicting

### Hence we try out the other model

######


###
####METHOD 2 - Logistic Regression Model
###



#Divide the data into Train and Test using stratified sample method

train.index = createDataPartition(santander_df$target, p = .75, list = FALSE)
train = santander_df[train.index,]
test = santander_df[-train.index,]

dim(train)
dim(test)

train$target<-as.factor(train$target)
test$target<-as.factor(test$target)


table(train$target)
table(test$target)
X_train = data.matrix(train[,-c(1,2)])
y_train_target = data.matrix(train$target)

X_test = data.matrix(test[, -c(1,2)])
y_test_traget = data.matrix(test$target)


#### 
#Logist_model = glm(target~., data = train, family = 'binomial')
#above method is giving error, it's througs memory error
####
# Hence we implement the GLMNET method




#Training the Random forest classifier with GLMNET (rationalization methond)

Logist_model = glmnet(X_train,y_train_target, family = 'binomial')

#Display the summer details
summary(Logist_model)

#Plot the graph on model ouput
plot(Logist_model)

print(Logist_model)

cross_val = cv.glmnet(X_train, y_train_target, family='binomial', type.measure = 'class')
plot(cross_val)



##Prediciton on Test data
Logist_predict = predict(cross_val, newx = X_test, type = "class", s='lambda.min')
Logist_predict


#Evaluate the performance of the classification model
dim(Logist_predict)
dim(y_test_traget)

class(Logist_predict)
class(y_test_traget)


target_df <- as.data.frame(y_test_traget)
target = as.factor(target_df$V1)
class(target)


Logist_predt<-as.data.frame(Logist_predict)
Predict_calss <- as.factor(Logist_predt$`1`)
class(Predict_calss)


confusionMatrix(data = Predict_calss, reference = target)

pred2<-as.integer(Logist_predict)
Test2<-as.factor(X_test)
Target2<-as.factor(y_test_traget)

## ROC Curve
roc_score2=roc(data=Test2, response=Target2, predictor=pred2, auc=TRUE, plot=TRUE)
plot(roc_score2)
print(roc_score2)


##Predict the Model with TEST DATASET
TEST<-as.matrix(test_df[,-c(1)])
colnames(TEST)

Test_Pred <- glmnet::predict.glmnet(Logist_model, newx = TEST, type="class")
Test_Pred

Test_Pred_Class<-ifelse(Test_Pred > 0.5, 1, 0)
summary(Test_Pred_Class)
table(Test_Pred_Class)


######
### Tried with GLMNET model with stratified sampling method, we can observer that Accuracy : 0.9145 and 
### AUC curve 0.6279, Model is predicting the 91 percent of accuracy and compare to Random forest model it's working
## quite well on unbalanced data

### Will try out the other model and see how it predicit the values

######



### 
#### Method 3 - NavieBayes model
###


#Split the training data using simple random sampling
train_index<-sample(1:nrow(santander_df),0.70*nrow(santander_df))
#train data
train_data<-santander_df[train_index,]
#validation data
test_data<-santander_df[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(test_data)


class(santander_df$target)


#Develop the model
NB_Model <- naiveBayes(target~., data=train_data[, -c(1)])

#predict on the test data
NB_Pred <- predict(NB_Model,test_data[, -c(1,2)], type = 'class')

#Evaluate the performance of the Naivebayes model
COnf_matrix = table(NB_Pred, test_data$target)
confusionMatrix(COnf_matrix)

table(NB_Pred)


X_test = data.matrix(test_data[, -c(1,2)])
y_test_traget = data.matrix(test_data$target)



pred3<-as.integer(NB_Pred)
Test3 <-as.factor(X_test)
Target3<-as.factor(y_test_traget)


## ROC Curve
roc_score3=roc(data=Test3, response=Target3, predictor=pred3, auc=TRUE, plot=TRUE)
plot(roc_score3)
print(roc_score3)



## Predict the Model on TEST dataset
Test_predict <- predict(NB_Model, test_df, type = 'class')
print(Test_predict)

table(Test_predict)


Test_Pred_DF<-data.frame(ID_CODE = test_df$ID_code, Predict_Class = Test_predict)
write.csv(Test_Pred_DF,"Santander_Test_Predict.csv",row.names = FALSE)
head(Test_Pred_DF)


######
### Tried with NAIVE BAYES model with simple random sampling method, we can observer that Accuracy : 0.9228 and 
### AUC curve 0.6758, Model is predicting the 92 percent of accuracy comparing with previous 2 model it's working
### quite well on unbalanced data


######

