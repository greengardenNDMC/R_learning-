#1. Packages--------------------------------------------------
library(caTools)
library(xgboost)
require(Matrix)
require(data.table)
library(caret)
if (!require('vcd')) install.packages('vcd')

#2. data preprocess--------------------------------------------------
data(Arthritis)
df <- data.table(Arthritis, keep.rownames = FALSE)
df

head(df[,AgeDiscret := as.factor(round(Age/10,0))])
head(df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))])
df[,ID:=NULL]
levels(df[,Treatment])

##依變項要是數值，ROSE資料插補用
##1有完全改善 0無完全改善
df$Improved<-ifelse(df$Improved=="Marked","1","0")
table(df$Improved) #0:56 1:28 


#3. split into training and testing sets----------------------
#Use 70% of dataset as training set and remaining 30% as testing set
##依據Improved欄位進行抽樣，讓兩組的Improved值分布類似
sample <- sample.split(df[,Improved], SplitRatio = 0.7)
train  <- subset(df, sample == TRUE)
test   <- subset(df, sample == FALSE)

dim(train) #59*6
dim(test) #30*29

#check dependent variable freq
table(train$Improved)
table(test$Improved)


#4. deal with imbalance data using ROSE------------------------
library("ROSE")
str(train)
train<- ROSE(Improved  ~ ., data = train, seed = 1)$data
table(train$Improved)


#5. Data transformation----------------------------------------
#data transformation to sparse matrix
sparse_matrix <- sparse.model.matrix(Improved ~ ., data = train)[,-1]
head(sparse_matrix)

#for testing set
sparse_matrix.t <- sparse.model.matrix(Improved ~ ., data = test)[,-1]
head(sparse_matrix.t)

#依變項label 
output_vector = train$Improved
output_vector

output_vector.t = test$Improved
output_vector.t

#6. XGboost 1--------------------------------
##nrounds 太多可能會overfitting要調整
#scale_pos_weight: for imbalance classification recommend value positive #/negative#  test result worse
# n_estimators: 樹的數量
# max_depth: 每顆樹的最大深度
# learning_rate: 範圍通常在0.01-0.2之間
# colsample_bytree：每次建樹可以使用多少比例的features

bst <- xgboost(data = sparse_matrix, label = output_vector, 
               max_depth = 5, # default 6 Typical values: 3-10
               eta = 1,
               colsample_bytree = 0.5,
               nthread = 2, 
               nrounds =300,
               scale_pos_weight =1,
               gamma = 0, #0-3
               subsample = 0.8,
               early_stopping_rounds = 2,
               objective = "binary:logistic")


##plot erro loss and iteration number
plot(bst$evaluation_log$iter,bst$evaluation_log$train_logloss)


##plot variable importance 
#變項重要性
importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
head(importance)
xgb.plot.importance(importance_matrix = importance)


##plot tree
xgb.plot.tree(model = bst, trees = 11) #plot final tree

##prediction.......................
preds = predict(bst,sparse_matrix.t)
pred <-  as.numeric(preds > 0.5)
caret::confusionMatrix(factor(pred),factor(output_vector.t))


##不同cut下sen spec變化..................
senspec=data.frame(Cut=1,Sen=1,Spec=1)
for(k in seq(0,1,0.1)){
  pred <-  as.numeric(preds < k)
  
  cfm=caret::confusionMatrix(as.factor(pred),as.factor(output_vector.t))
  cfm2=data.frame(cfm$byClass)
  message("cut:",k, " Sen:",cfm2[1,],"   Spec:",cfm2[2,])
  senspec[nrow(senspec) + 1,] = c(k,cfm2[1,], cfm2[2,])
}

senspec=senspec[-1,] #delete first row

##plot cut, sen, spec
library(reshape2)
mlt=melt(senspec,
         id="Cut",
         variable.name="Indicator",
         value.name="Value")

ggplot(mlt,aes(x=Cut,y=Value,group=Indicator,color=Indicator))+
  geom_line()


#最佳切點為0.26左右
preds = predict(bst,sparse_matrix.t)
pred <-  as.numeric(preds > 0.26)
caret::confusionMatrix(factor(pred),factor(output_vector.t))

#7. Cross-validation XGboost--------------------------------
##find the best parameters
##objective: multi:softprob 為多分類 
##因為參數組合多，迴圈次數最好多一點，比較可以找到最佳參數組合
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:1000) {
  param <- list(objective = "binary:logistic",
                metrics = list("rmse","auc","aucpr","logloss"),
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1),
                prediction = TRUE
  )
  cv.nround = 100
  cv.nfold = 3
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data = sparse_matrix, label =output_vector, 
                 params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log$test_logloss_mean)
  min_logloss_index = which.min(mdcv$evaluation_log$test_logloss_mean)
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

#獲得最佳參數
best_param

#最佳迴圈數
nround = best_logloss_index
nround

#最佳隨機種子
best_seednumber
set.seed(best_seednumber)


bst.cv=xgboost(params = best_param,
               data = sparse_matrix, 
               label = output_vector,
               nrounds=nround,
               early_stopping_rounds = 8
               )

##plot error loss and iteration number
plot(bst.cv$evaluation_log$iter,bst.cv$evaluation_log$train_logloss)


##plot variable importance 
#變項重要性
importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst.cv)
head(importance)
xgb.plot.importance(importance_matrix = importance)


##prediction.......................
preds = predict(bst.cv,sparse_matrix.t)
##不同cut下sen spec變化..................
senspec=data.frame(Cut=1,Sen=1,Spec=1)
for(k in seq(0,1,0.1)){
  pred <-  as.numeric(preds < k)
  
  cfm=caret::confusionMatrix(as.factor(pred),as.factor(output_vector.t))
  cfm2=data.frame(cfm$byClass)
  message("cut:",k, " Sen:",cfm2[1,],"   Spec:",cfm2[2,])
  senspec[nrow(senspec) + 1,] = c(k,cfm2[1,], cfm2[2,])
}

senspec=senspec[-1,] #delete first row

##plot cut, sen, spec
library(reshape2)
mlt=melt(senspec,
         id="Cut",
         variable.name="Indicator",
         value.name="Value")

ggplot(mlt,aes(x=Cut,y=Value,group=Indicator,color=Indicator))+
  geom_line()


#最佳切點為0.37左右
preds = predict(bst,sparse_matrix.t)
pred <-  as.numeric(preds > 0.37)
caret::confusionMatrix(factor(pred),factor(output_vector.t))


##plot tree
xgb.plot.tree(model = bst, trees = 10) #plot final tree


