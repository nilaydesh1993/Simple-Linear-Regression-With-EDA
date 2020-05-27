# =============================================================================
# Business Problem :- Build a prediction model for Salary_hike
# =============================================================================
Salary_Data= read.csv(file.choose())
colnames(Salary_Data) <- c("YE","SH")
View(Salary_Data)

################################## - EDA - ###################################

#Measures of Central Tendency / First moment business decision
attach(Salary_Data) 
# Mean
mean(SH)
mean(YE)
# Median
median(SH)
median(YE)
# Mode
table1 = table(SH)
table1
table1[table1 == max(table1)]
table2 = table(YE)
table2
table2[table2 == max(table2)]

# Measures of Dispersion / Second moment business decision
# variance
var(SH) 
var(YE)
#standard deviation
sd(SH) 
sd(YE)

library(moments)
#Third moment business decision
skewness(SH)
skewness(YE)

#Fourth moment business decision
kurtosis(SH)
kurtosis(YE)

#Graphical Representation
#histogram
hist(SH)
hist(YE)
#boxplot
boxplot(SH)
boxplot(YE)
#Normal Quantile-Quantile Plot
qqnorm(YE)
qqline(YE)
qqnorm(SH)
qqline(SH)
#summary
summary(Salary_Data)

########################### - Liner Regression Model - ########################
# X = YEAR OF EXPERIANCE , Y = SALARY HIKE 

# scatter plot
plot(YE,SH) # Strong positive correlation

# correlation coefficient (r)
cor(YE,SH) # # r = 0.98 - Strong Correlation

# linear regression model
model_q4 <- lm(SH ~ YE) 
summary(model_q4) 
### Bo - 25792 , B1 = =9449.96 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.96 - Strong Correlation

predict(model_q4)
model_q4$residuals
sum(model_q4$residuals)  # sum of error is equal to 0

confint(model_q4,level=0.95) #final model
predict(model_q4, interval = "confidence")

rmse <- sqrt(mean(model_q4$residuals^2))
rmse #5592

library(ggplot2)
ggplot(data = Salary_Data,aes(x=YE,y=SH))+
  geom_point(colour = 'blue')+
  geom_line(colour = 'red',data = Salary_Data,aes(x=YE , y = predict(model_q4)))

"""
Final Equation -
Salary_hike = 9449.96(year of experiance) + 25790

With Confidance interval 95% 
Salary_hike = 8674.11(year of experiance) + 21100
Salary_hike = 10200(year of experiance) + 30400"""