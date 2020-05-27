# =============================================================================
# Business Problem :- Predict weight gained using calories consumed.
# =============================================================================
library(readr)
calories_consumed = read.csv(file.choose())
colnames(calories_consumed) <- c("gained","calories")
View(calories_consumed)

################################## - EDA - ###################################

#Measures of Central Tendency / First moment business decision
attach(calories_consumed) 
# Mean
mean(gained)
mean(calories)
# Median
median(gained)
median(calories)
# Mode
table1 = table(gained)
table1
table1[table1 == max(table1)]
table2 = table(calories)
table2
table2[table2 == max(table2)]

# Measures of Dispersion / Second moment business decision
# variance
var(gained) 
var(calories)
#standard deviation
sd(gained) 
sd(calories)

library(moments)
#Third moment business decision
skewness(gained)
skewness(calories)

#Fourth moment business decision
kurtosis(gained)
kurtosis(calories)

#Graphical Representation
#histogram
hist(gained)
hist(calories)
#boxplot
boxplot(gained)
boxplot(calories)
#Normal Quantile-Quantile Plot
qqnorm(calories)
qqline(calories)
qqnorm(gained)
qqline(gained)
#summary
summary(calories_consumed)

########################### - Liner Regression Model - ########################
# X = Calories Consume , Y = Weigth gain 

# scatter plot
plot(calories,gained) # Strong positive correlation

# correlation coefficient (r)
cor(calories,gained) # r = 0.94

# linear regression model
model_q1 <- lm(gained ~ calories) 
summary(model_q1) 
# R2 = 0.897 Strong correlation
# B0 = -625.75 , B1 = 0.42 Pvalue are Significants can used this value for model building

predict(model_q1)
model_q1$residuals
sum(model_q1$residuals)  # sum of error is equal to 0

confint(model_q1,level=0.95) #final model
predict(model_q1, interval = "confidence")

rmse <- sqrt(mean(model_q1$residuals^2))
rmse #103.3

library(ggplot2)
ggplot(data = calories_consumed,aes(x=calories,y=gained))+
  geom_point(colour = 'blue')+
  geom_line(colour = 'red',data = calories_consumed,aes(x=calories , y = predict(model_q1)))

"""
Final Equation -
gained = 0.42(carories) - 625.75

With Confidance interval 95% 
gained = 0.33(calories) - 845.42
gained = 0.50(calories) - 406.42 """
