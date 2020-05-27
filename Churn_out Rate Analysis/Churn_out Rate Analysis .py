#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:12:51 2020
@author: DESHMUKH
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab
import statsmodels.formula.api as smf

# =============================================================================
# Business Problem :- Build a prediction model for Churn_out_rate 
# =============================================================================

emp_data = pd.read_csv("emp_data.csv")
emp_data.columns = "SH","COR"

################################## - EDA - ###################################

# First moment business Decision / Measure of central tendancy
emp_data.mean()
emp_data.median()
emp_data.mode()
emp_data.SH.value_counts()
emp_data.COR.value_counts()

# Second moment business Decision / Measure of Dispersion
emp_data.var()
emp_data.std()

# Third and Forth business Decision / Skewness and Kurtosis
emp_data.skew()
emp_data.kurt()

# Graphical represention 
## Histogram 
plt.hist(emp_data.SH,color = 'red') # Rigth Skewed
plt.hist(emp_data.COR,color = 'green') # Rigth Skewed

## Boxplot
plt.boxplot(emp_data.SH) # no Outliers
plt.boxplot(emp_data.COR) # no Outliers

# Normal Quantile Quantile plot
stats.probplot(emp_data.SH,dist = 'norm',plot = pylab) # Data are assumed not normal
stats.probplot(emp_data.COR,dist = 'norm',plot = pylab) # Data are assumed not normal

# Summary
emp_data.describe()

########################### - Liner Regression Model - ########################
# X = Salary hike , Y = Churn out rate

# Scatter plot
plt.scatter(x = emp_data.SH, y = emp_data.COR, color = 'purple')

# Correlation coifficent (r)
np.corrcoef(emp_data.SH,emp_data.COR) # r = -0.91 - Strong Correlation but curvilinear in nature

# Linear Regression Model
model_q3 = smf.ols('emp_data.COR~emp_data.SH',data = emp_data).fit()
model_q3.summary() 
## Bo - 244.36 , B1 = -0.1015 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.83 - Strong Correlation

# Prediction
predict_q3 = model_q3.predict(pd.DataFrame(emp_data.SH))
predict_q3

# Errors
res_q3 = emp_data.COR - predict_q3
np.sum(res_q3)  # sum of error aprroximatly 0

# RMSE
sqr_q3 = res_q3 * res_q3
mes_q3 = np.mean(sqr_q3)
rmse_q3 = np.sqrt(mes_q3)
rmse_q3 # 3.99

# Final representation
plt.scatter(emp_data.SH,emp_data.COR, color = 'red')
plt.plot(emp_data.SH,predict_q3)
plt.show()

####################- Logarithmic Linear Regression Model -#####################
# X = log(Salary hike) , Y = Churn out rate

# Scatter plot
plt.scatter(x = np.log(emp_data.SH), y = emp_data.COR, color = 'red')

# Correlation coifficent (r)
np.corrcoef(np.log(emp_data.SH),emp_data.COR) # r = -0.92 - Strong Correlation

# Linear Regression Model (log)
model_q3_log = smf.ols('emp_data.COR~np.log(emp_data.SH)',data = emp_data).fit()
model_q3_log.summary() 
## Bo - 1381.45 , B1 = -176.11 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.85 - Strong Correlation

# Prediction
predict_q3_log = model_q3_log.predict(pd.DataFrame(emp_data.SH))
predict_q3_log

# Errors
res_q3_log = emp_data.COR - predict_q3_log
np.sum(res_q3_log)  # sum of error aprroximatly 0

# RMSE
sqr_q3_log = res_q3_log * res_q3_log
mes_q3_log = np.mean(sqr_q3_log)
rmse_q3_log = np.sqrt(mes_q3_log)
rmse_q3_log # 3.78

####################- Exponential Regression Model -#####################
# X = Salary hike , Y = Log(Churn out rate)

# Scatter plot
plt.scatter(x = emp_data.SH, y = np.log(emp_data.COR), color = 'blue')

# Correlation coifficent (r)
np.corrcoef(emp_data.SH,np.log(emp_data.COR)) # r = -0.93 - Strong Correlation

# Linear Regression Model (log)
model_q3_expo = smf.ols('np.log(emp_data.COR)~emp_data.SH',data = emp_data).fit()
model_q3_expo.summary() 
## Bo = 6.63 , B1 = -0.0014 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.87 - Strong Correlation

# Prediction
predict_q3_expo= model_q3_expo.predict(pd.DataFrame(emp_data.SH))
predict_q3_expo
predict_q3_ext = np.exp(predict_q3_expo)   # Re transformation as output should not have in sqrt
predict_q3_ext

# Errors
res_q3_expo = emp_data.COR - predict_q3_ext
np.sum(res_q3_expo)  # sum of error aprroximatly 1

# RMSE
sqr_q3_expo = res_q3_expo * res_q3_expo
mes_q3_expo = np.mean(sqr_q3_expo)
rmse_q3_expo = np.sqrt(mes_q3_expo)
rmse_q3_expo  # 3.54

# Final representation
plt.scatter(x = emp_data.SH, y = np.log(emp_data.COR), color = 'blue')
plt.plot(emp_data.SH,predict_q3_expo)
plt.show()

####################- Polynomial Linear Regression Model -#####################
# X = Polynomial(Salary hike) , Y = Churn out rate

# Correlation coifficent (r)
np.corrcoef((emp_data.SH**2),emp_data.COR) # r = -0.90 - Strong Correlation

# Linear Regression Model (polynomial)
model_q3_poly = smf.ols('emp_data.COR~emp_data.SH + I(emp_data.SH**2)',data = emp_data).fit()
model_q3_poly.summary()
## Bo = 1647.01 , B1 = -1.7371 , B2 = 0.0005 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.97 - Strong Correlation

# Prediction
predict_q3_poly = model_q3_poly.predict(pd.DataFrame(emp_data.SH))
predict_q3_poly # Prediction

# Errors
res_q3_poly = emp_data.COR - predict_q3_poly
np.sum(res_q3_poly)  # sum of error aprroximatly 0

# RMSE
sqr_q3_poly = res_q3_poly**2
mes_q3_poly = np.mean(sqr_q3_poly)
rmse_q3_poly = np.sqrt(mes_q3_poly)
rmse_q3_poly  # 1.57

# Final representation
plt.scatter(emp_data.SH,emp_data.COR, color = 'blue')
plt.plot(emp_data.SH,predict_q3_poly,color = 'black')
plt.show()

"""
Final Equation -
Churn_out_rate  = 1647.01 - 1.7371(Salary hike) + 0.0005(Salary hike)^2

With Confidance interval 95% 
gained = 1107.8 - 2.365(Salary hike) + 0.000(Salary hike)^2
gained = 2186.3 - 1.109(Salary hike) + 0.001(Salary hike)^2 """




