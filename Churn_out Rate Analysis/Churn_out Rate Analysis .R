# =============================================================================
# Business Problem :- Build a prediction model for Churn_out_rate 
# =============================================================================
library(readr)
emp_data = read.csv(file.choose())
colnames(emp_data) <- c("SH","COR")
View(emp_data)
attach(emp_data) 

################################## - EDA - ###################################

#Measures of Central Tendency / First moment business decision
# Mean
mean(COR)
mean(SH)
# Median
median(COR)
median(SH)
# Mode
table1 = table(COR)
table1
table1[table1 == max(table1)]
table2 = table(SH)
table2
table2[table2 == max(table2)]

# Measures of Dispersion / Second moment business decision
# variance
var(COR) 
var(SH)
#standard deviation
sd(COR) 
sd(SH)

library(moments)
#Third moment business decision
skewness(COR)
skewness(SH)

#Fourth moment business decision
kurtosis(COR)
kurtosis(SH)

#Graphical Representation
#histogram
hist(COR)
hist(SH)
#boxplot
boxplot(COR)
boxplot(SH)
#Normal Quantile-Quantile Plot
qqnorm(SH)
qqline(SH)
qqnorm(COR)
qqline(COR)
#summary
summary(emp_data)

########################### - Liner Regression Model - ########################
# X = SALARY HIKE , Y = CHURN OUT RATE 

# Scatter plot
plot(SH,COR) # Strong negative correlation

# Correlation coefficient (r)
cor(SH,COR) # r = -0.91 - Strong Correlationbut curvilinear in nature

# Linear regression model
model_q3 <- lm(COR ~ SH) 
summary(model_q3) 
## Bo - 244.36 , B1 = -0.1015 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.83 - Strong Correlation

# Prediction
predict(model_q3)
model_q3$residuals
sum(model_q3$residuals)  # sum of error is equal to 0

confint(model_q3,level=0.95) #final model
predict(model_q3, interval = "confidence")

# Rmse
rmse <- sqrt(mean(model_q3$residuals^2))
rmse #3.99

# ggplot
library(ggplot2)
ggplot(data = emp_data,aes(x=SH,y=COR))+
  geom_point(colour = 'blue')+
  geom_line(colour = 'red',data = emp_data,aes(x=SH , y = predict(model_q3)))

####################- Polynomial Linear Regression Model -#####################
# X = Polynomial(Salary hike) , Y = Churn out rate

# correlation coefficient (r)
cor(SH**2,COR) ## r = -0.90 - Strong Correlation

# polynimial linear regression model
model_q3poly <- lm(COR ~ SH + I(SH * SH))
summary(model_q3poly)
## Bo = 1647.01 , B1 = -1.7371 , B2 = 0.0005 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.97 - Strong Correlation

# Prediction
predict(model_q3poly)
model_q3poly$residuals
sum(model_q3poly$residuals)  # sum of error is equal to 0

confint(model_q3poly,level=0.95)
predict(model_q3poly,interval="confidence")

# RMSE
rmseq3poly <- sqrt(mean(model_q3poly$residuals^2))
rmseq3poly# 1.57

# ggplot
ggplot(data = emp_data,aes(x=SH,y=COR))+
  geom_point(colour = 'blue')+
  geom_line(colour = 'red',data = emp_data,aes(x=SH , y = predict(model_q3poly)))

"""
Final Equation -
Churn_out_rate  = 1647.01 - 1.7371(Salary hike) + 0.0005(Salary hike)^2

With Confidance interval 95% 
gained = 1107.8 - 2.365(Salary hike) + 0.000(Salary hike)^2
gained = 2186.3 - 1.109(Salary hike) + 0.001(Salary hike)^2 """