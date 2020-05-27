#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:28:23 2020
@author: DESHMUKH
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab
import statsmodels.formula.api as smf

# =============================================================================
# Business Problem :- Build a prediction model for Salary_hike
# =============================================================================

Salary_Data = pd.read_csv("Salary_Data.csv")
Salary_Data.columns = "YE","SAL"

################################## - EDA - ###################################

# First moment business Decision / Measure of central tendancy
Salary_Data.mean()
Salary_Data.median()
Salary_Data.mode()
Salary_Data.YE.value_counts()
Salary_Data.SAL.value_counts()

# Second moment business Decision / Measure of Dispersion
Salary_Data.var()
Salary_Data.std()

# Third and Forth business Decision / Skewness and Kurtosis
Salary_Data.skew()
Salary_Data.kurt()

# Graphical represention 
## Histogram 
plt.hist(Salary_Data.YE,color = 'red') # Rigth Skewed
plt.hist(Salary_Data.SAL,color = 'green') # Rigth Skewed

## Boxplot
plt.boxplot(Salary_Data.YE) # no Outliers
plt.boxplot(Salary_Data.SAL) # no Outliers

# Normal Quantile Quantile plot
stats.probplot(Salary_Data.YE,dist = 'norm',plot = pylab) # Data are assumed not normal
stats.probplot(Salary_Data.SAL,dist = 'norm',plot = pylab) # Data are assumed not normal

#summary
Salary_Data.describe()

########################### - Liner Regression Model - ########################
# Y = Salary hike , X = Year of experiance 

# Scatter plot
plt.scatter(x = Salary_Data.YE, y = Salary_Data.SAL, color = 'green') 

# Correlation coifficent (r)
np.corrcoef(x = Salary_Data.YE, y = Salary_Data.SAL) # r = 0.98 - Strong Correlation

# Linear Regression Model
model_q4 = smf.ols('Salary_Data.SAL ~ Salary_Data.YE',data = Salary_Data).fit()
model_q4.summary()
model_q4.params 
### Bo - 25792.2 , B1 = =9449.96 - Pvalue is less than 0.05 so we can use this data for model bulding
## R2 = 0.96 - Strong Correlation

predict_q4 = model_q4.predict(pd.DataFrame(Salary_Data.YE))
predict_q4

res_q4 = Salary_Data.SAL - predict_q4
np.sum(res_q4)  # Sum of error aprroximatly 0

# RMSE
sqr_q4 = res_q4 * res_q4
mes_q4 = np.mean(sqr_q4)
rmse_q4 = np.sqrt(mes_q4)
rmse_q4 # 5592

# Final representation
plt.scatter(x = Salary_Data.YE, y = Salary_Data.SAL, color = 'blue')
plt.plot(Salary_Data.YE,predict_q4,color = 'black')
plt.show()


"""
Final Equation -
Salary_hike = 9449.96(year of experiance) + 25790

With Confidance interval 95% 
Salary_hike = 8674.11(year of experiance) + 21100
Salary_hike = 10200(year of experiance) + 30400"""

