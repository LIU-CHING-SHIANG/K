rm(list=ls(all=TRUE))

library(glmnet)
library(forecast)
library(lmtest)
library(tseries)
library(zoo)
library(stargazer)
library(broom)
macrorow <- read.csv("MacroData.csv", stringsAsFactors = FALSE)
MacroData<-macrorow[complete.cases(macrorow), ]
sum(is.na(MacroData))

str(MacroData)
MacroData[, -1] <- lapply(MacroData[, -1], as.numeric)
str(MacroData)

split_point <- floor(0.985 * nrow(MacroData))

train <- 1:split_point
test <- (split_point + 1):nrow(MacroData)

半導以外 <- MacroData[, !names(MacroData) %in% c("市半導體","日期")]
日期以外 <- MacroData[, !names(MacroData) %in% "日期"]
tail(MacroData[train,]) #檢查樣本用

##時間序列
y_ts <- ts(日期以外[train,]$市半導體, frequency = 12, start = c(2011,01))
reg_model <- tslm(y_ts ~ .,data = 半導以外[train,])
summary(reg_model)


##逐步回歸
base <- glm(市半導體~1, family=gaussian,
            data=日期以外[train,])
summary(base)


full <-glm(市半導體~., family=gaussian,
           data=日期以外[train,]) 
summary(full)

fwdBIC <- step(base, scope=formula(full), 
               direction="forward", k=log(length(train)),
               trace=0)
summary(fwdBIC)

#LASSO
X<-model.matrix(formula(full),日期以外)
X<-X[,-1]

cvfit<-cv.glmnet(x =X[train,], y =日期以外$市半導體[train],   #cv 交叉驗證
                 family="gaussian",
                 alpha=1, standardize=TRUE)
betas<-coef(cvfit, s =2)
model.1se <-which(betas[2:length(betas)]!=0)


colnames(X[,model.1se])


##逐步預測
predBIC <- predict(fwdBIC, newdata=日期以外[test,],
                   type="response")

##時序預測
forecast_result <- forecast(reg_model, newdata = 日期以外[test,])
summary(forecast_result)

#所有變數丟進去
refitting.data <- data.frame(日期以外$市半導體, X[,model.1se])
names(refitting.data)[1] <- "市半導體"
post.lasso.1se <- glm(市半導體 ~ ., family="gaussian",
                      data=refitting.data[train,])
#LASSO預測
predLasso.1se <- predict(post.lasso.1se,
                         newdata=refitting.data[test,],
                         type="response")


#主成分分析
pcmacro <- prcomp(日期以外, scale=TRUE)
round(pcmacro$rotation, 1)
pcmacro$rotation
round(predict(pcmacro, newdata=日期以外), 2)
plot(pcmacro, main="")
mtext(side=1,"MacroData PrincipleComponents", line=1, font=2)
summary(pcmacro)

screeplot(pcmacro, type = "lines")
eigenvalues <- pcmacro$sdev^2
num_factors_kaiser <- sum(eigenvalues > 1)
print(paste("根据Kaiser准则选择的主成分数量:", num_factors_kaiser))

#PC建立GLM
pc_data <- as.data.frame(pcmacro$x[, 1:6])
pc_data$市半導體 <- 日期以外$市半導體
pc_model <- glm(pc_data[train,]$市半導體 ~ .,family=gaussian, data = pc_data[train,])


#PC預測
predpc <- predict(pc_model, newdata=pc_data[test,],
                   type="response")


##FULL參考用 會刪掉
predfull <- predict(full, newdata=日期以外[test,],
                    type="response")
errorFull <- (日期以外[test,1]-predfull)^2
rmseFull <- sqrt(mean(errorFull))
rmseFull
## 到這裡都是參考用


errorBIC <- (日期以外[test,1]-predBIC)^2
rmseBIC <- sqrt(mean(errorBIC))  ##用RMSE貼近資料型態

errorLasso.1se <- (日期以外[test,1] - predLasso.1se)^2 
rmseLasso <- sqrt(mean(errorLasso.1se))

rmseTS <- sqrt(mean((日期以外[test,]$市半導體 - forecast_result$mean)^2))

rmsePC <- sqrt(mean((pc_data[test,]$市半導體 - predpc)^2))

c(BIC=rmseBIC, LASSO=rmseLasso, TS=rmseTS, PC=rmsePC)


predBIC

forecast_result


