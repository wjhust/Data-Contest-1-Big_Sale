library(ggplot2)
library(grid)
library(gridExtra)
library(caret)
library(randomForest)
library(xgboost)
library(Hmisc)
library(dplyr)
library(rpart)
library(rpart.plot)

##数据读取
train.data<- read.csv("/Users/wangjian/Desktop/R/test1/train.csv",stringsAsFactors = F)
View(train.data)
test.data<-read.csv("/Users/wangjian/Desktop/R/test1/test.csv",stringsAsFactors = F)
View(test.data)
test.data$Item_Outlet_Sales<-NA#设置test数据的响应变量为NA
View(test.data)
all_bms<-rbind(train.data,test.data)#设置为全局，方便进行训练
View(all_bms)
describe(all_bms)#描述性统计分析


##数据清洗1---weight缺失值均值填充
tmp<-aggregate(Item_Weight~Item_Identifier,data=all_bms,FUN = mean)

#对同一个商品id下的各个商品统计重量得均值（同一个商品：比如乐事，重量应该是相同的）
View(tmp)
#把商品均值填充到同一个类别下得各个商品得NA
for (i in which(is.na(all_bms$Item_Weight))){
  all_bms$Item_Weight[i]<-tmp$Item_Weight[tmp$Item_Identifier==all_bms$Item_Identifier[i]]
}
sum(is.na(all_bms$Item_Weight))#再次查看商品weight得缺失值


#清洗2--商店大小的缺失值Size均值填充
prop.table(table(all_bms$Outlet_Size))#将表中得条目表示为分数形式
tmp2<-aggregate(Item_Outlet_Sales~Outlet_Identifier+Outlet_Type+Outlet_Size,data = all_bms,FUN = mean)
tmp2
#查看平均售价（响应变量）和商店类型得关系
#使用决策树进行填补商店类型得缺失
fit<-rpart(factor(Outlet_Size)~Outlet_Type,data=all_bms[all_bms$Outlet_Size!="",],method = "class")
pred<-predict(fit,all_bms[all_bms$Outlet_Size=="",],type="class")
all_bms$ Outlet_Size[all_bms$Outlet_Size==""]<-as.vector(pred)
table(all_bms$Outlet_Size,all_bms$Outlet_Identifier)



##特征工程
all_bms$Item_Sales_Vol<-round(all_bms$Item_Outlet_Sales/all_bms$Item_MRP+0.5,0)
#创建-销量-作为预测变量
View(all_bms$Item_Sales_Vol)
summary(all_bms$Item_Fat_Content)
#处理含脂肪类型，合并类别
all_bms$Item_Fat_Content<-as.character(all_bms$Item_Fat_Content)
all_bms$Item_Fat_Content[all_bms$Item_Fat_Content%in%c("LF","low fat")]<-"Low Fat"
all_bms$Item_Fat_Content[all_bms$Item_Fat_Content%in%c("reg")]<-"Regular"
table(all_bms$Item_Fat_Content)
#进一步归类商品的类别
summary(all_bms$Item_Type)
all_bms$Item_Attribute<-factor(substr(all_bms$Item_Identifier,1,2))
View(all_bms$Item_Attribute)
#提取ID前两个字母作为商品的类别
table(all_bms$Item_Attribute)
View(all_bms$Item_Attribute)
all_bms$Item_Fat_Content[all_bms$Item_Attribute=="NC"]<-"Non-Food"
#非食物类是没有低脂高脂的情况
table(all_bms$Item_Fat_Content)
#这里整体新建了变量attribute：DR为饮品，DF为食物，NC为消耗品三个类别
#由于存在非消耗品，这里对Item_Fat_Content增加了一个因子NonFood,对应非食品类的脂肪情况
View(all_bms$Item_Fat_Content)


##异常数处理：Item_Visibility中为0的数据---均值填充
tmp3<-aggregate(Item_Visibility~Outlet_Identifier,data=all_bms,FUN=mean)
View(tmp3)
z<-data.frame(tmp3$Outlet_Identifier,tmp3$Item_Visibility)
plot(z)
mean(tmp3$Item_Visibility)
p<-(0.05900006+0.05997634+0.06018442+0.06024158+0.06034360+0.06082599+0.06090731+0.06114207)/8
p
for (i in which(all_bms$Item_Visibility==0)){
  all_bms$Item_Visibility[i]<-p
}
sum(all_bms$Item_Visibility)


##增加新的时间变量--Outlet_Years
View(all_bms$Outlet_Establishment_Year)
all_bms$Outlet_Years<-(2013-all_bms$Outlet_Establishment_Year)
View(all_bms$Outlet_Years)
#新变量Outlet_Years用来保存每个商店成立的年数，为离散型变量，准备使用因子类型


##将所有分类变量转化为因子，消除不存在的因子
cols<-c("Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Type","Outlet_Years","Item_Attribute","Outlet_Identifier") 
   for (i in cols){
     all_bms[,i]<-factor(all_bms[,i])
   } 


##可视化探索变量
#商店层面：
p1<-data.frame(all_bms$Outlet_Type,all_bms$Item_Sales_Vol)
View(p1)
plot(p1)
p2<-data.frame(all_bms$Outlet_Location_Type,all_bms$Item_Sales_Vol)
plot(p2)
p3<-data.frame(all_bms$Outlet_Years,all_bms$Item_Sales_Vol)
plot(p3)
p4<-data.frame(all_bms$Outlet_Size,all_bms$Item_Sales_Vol)
plot(p4)
#商品层面：
w1<-data.frame(all_bms$Item_Visibility,all_bms$Outlet_Type,all_bms$Item_Sales_Vol)
View(w1)
plot(w1)
w2<-data.frame(all_bms$Item_Attribute,all_bms$Item_Sales_Vol)
plot(w2)
w3<-data.frame(all_bms$Item_Fat_Content,all_bms$Item_Sales_Vol)
plot(w3)
w4<-data.frame(all_bms$Item_Type,all_bms$Item_Sales_Vol)
plot(w4)


## 建模：
train<-all_bms[!is.na(all_bms$Item_Outlet_Sales),]
View(train)
test<-all_bms[is.na(all_bms$Item_Outlet_Sales),]
View(test)
set.seed(1234)
ind<-createDataPartition(train$Item_Sales_Vol,p=0.7,list = FALSE)
train_val<-train[ind,]
test_val<-train[-ind,]
View(train_val)
View(all_bms)

#决策树
myformula <- Item_Sales_Vol ~ Outlet_Location_Type+Outlet_Type+ Outlet_Years+ Outlet_Size+ Item_Visibility + Item_MRP + Item_Type
#创建模型评估RMSE函数 
model.rmse <- function(pred,act){ sqrt(sum((act - pred)^2)/length(act)) }
fit.tr <- rpart(myformula,
                data = train_val,
                method = "anova")
summary(fit.tr)
rpart.plot(fit.tr)
pred <- predict(fit.tr,test_val) 
model.rmse(pred*test_val$Item_MRP,test_val$Item_Outlet_Sales)
pred.test <- predict(fit.tr,test)
submit <- data.frame(Item_Identifier = test.data$Item_Identifier, Outlet_Identifier = test.data$Outlet_Identifier,Item_Outlet_Sales=pred.test*test$Item_MRP)
write.csv(submit, file = "dtree3.csv", row.names = FALSE)


#随机森林--由于有characte不能使用
set.seed(2345) 
View(train_val)
fit.rf <- randomForest(myformula,data = train_val, ntree=1000)
summary(fit.rf)
pred <- predict(fit.rf,test_val)
model.rmse(pred*test_val$Item_MRP,test_val$Item_Outlet_Sales)
#创建用于上传评分的测试结果
pred.test <- predict(fit.rf,test)
submit <- data.frame(Item_Identifier = test.data$Item_Identifier, Outlet_Identifier = test.data$Outlet_Identifier,Item_Outlet_Sales=pred.test*test$Item_MRP) 
write.csv(submit, file = "rf2.csv", row.names = FALSE)


#gbm
Ctrl <- trainControl(method="repeatedcv",number=6,repeats=5)
set.seed(3456)
fit.gbm <- train(myformula,
                 data = train_val,
                 trControl=Ctrl,
                 method="gbm",
                 verbose=FALSE
) 
summary(fit.gbm) 
pred <- predict(fit.gbm,test_val)
model.rmse(pred*test_val$Item_MRP,test_val$Item_Outlet_Sales)
pred.test <- predict(fit.gbm,test)
submit <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier,Item_Outlet_Sales=pred.test*test$Item_MRP) 
write.csv(submit, file = "gbm-2cv.csv", row.names = FALSE)



##xgboost--要求所有变量为数值型
#1.构建稀疏矩阵
mymatrix <- function(train){
  matrix_num <- train[,c("Item_Visibility","Item_MRP")]
  matrix_num <- cbind(matrix_num,
                      model.matrix(~Outlet_Type-1,train),
                      model.matrix(~Outlet_Location_Type-1,train),
                      model.matrix(~Outlet_Size-1,train),
                      model.matrix(~Item_Type-1,train),
                      model.matrix(~Outlet_Years-1,train)  
  )
  return(data.matrix(matrix_num))
}
#获取每个数据集的稀疏矩阵
xgb.train_val <- mymatrix(train_val)
xgb.test_val <- mymatrix(test_val) 
xgb.test <- mymatrix(test)
#生成xgboost模型的D矩阵
dtrain_val <- xgb.DMatrix(data =xgb.train_val,label=train_val$Item_Sales_Vol)
dtest_val <- xgb.DMatrix(data = xgb.test_val,label=test_val$Item_Sales_Vol)
dtest_sub <- xgb.DMatrix(data = xgb.test)
#建模：
s=6.76

model<-xgboost(data=dtrain_val,nrounds = 12,max.depth=5)
pred <- predict(model,dtest_val)
model.rmse(pred*test_val$Item_MRP,test_val$Item_Outlet_Sales)
xgb.importance(colnames(xgb.train_val),model)
pred.test <- predict(model,dtest_sub)
submit <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier,Item_Outlet_Sales=pred.test*test$Item_MRP)
write.csv(submit, file = "xgb12-5.csv", row.names = FALSE)
