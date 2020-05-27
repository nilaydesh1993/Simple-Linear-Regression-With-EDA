import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pylab as plt
import pylab

# =============================================================================
# Business Problem :- Predict weight gained using calories consumed.
# =============================================================================

calories_consumed = pd.read_csv("calories_consumed.csv")
calories_consumed

calories_consumed.columns ="gained","calories"

################################## - EDA - ###################################

# Measures of Central Tendency / First moment business decision
calories_consumed.mean()
calories_consumed.median()
calories_consumed.mode()

# Measures of Dispersion / Second moment business decision
calories_consumed.var()
calories_consumed.std()

# Skewnees / Third moment business decision
calories_consumed.skew()

# Kurtosis / Fourth moment business decision 
calories_consumed.kurt()

# Graphical Representaion  
## Histogram
plt.hist(calories_consumed.gained) # Rigth skeweed
plt.hist(calories_consumed.calories) # Rigth skeweed

## Boxplot
plt.boxplot(calories_consumed.gained)
plt.boxplot(calories_consumed.calories)

# Normal Quantile-Quantile plot
stats.probplot(calories_consumed.gained,dist = 'norm',plot = pylab)
stats.probplot(calories_consumed.calories,dist = 'norm',plot = pylab)

calories_consumed.describe()

############################# - Scatter plot - ###############################

plt.scatter(calories_consumed.calories, calories_consumed.gained,color = 'green') 
# Strong positive correlation

######################## - Correlation Coefficient - ##########################

np.corrcoef(calories_consumed.calories,calories_consumed.gained)
# r = 0.94 Strong Positive correlation 

########################### - Liner Regression Model - ########################
# X = Calories Consume , Y = Weigth gain 

model_q1 = smf.ols('calories_consumed.gained~calories_consumed.calories',data = calories_consumed).fit()
model_q1.summary()
# R2 = 0.897 Strong correlation
# B0 = -625.75 , B1 = 0.42 Pvalue are Significants can used this value for model building

# Prediction
Predict_q1 = model_q1.predict(pd.DataFrame(calories_consumed.calories))
Predict_q1 

# Error
res_q1 = calories_consumed.gained - Predict_q1 
np.sum(res_q1) # sum of error is equal to 0

# RMSE 
squ_q1 = res_q1 * res_q1
mes_q1 = np.mean(squ_q1)
rmse_q1 = np.sqrt(mes_q1)
rmse_q1 # 103.3

plt.scatter(calories_consumed.calories, calories_consumed.gained,color = 'blue') 
plt.plot(calories_consumed.calories, Predict_q1 ,color = 'black')
plt.show()

"""
Final Equation -
gained = 0.42(carories) - 625.75

With Confidance interval 95% 
gained = 0.33(calories) - 845.42
gained = 0.50(calories) - 406.42 """


