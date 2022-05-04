

# import regular libraries
from tkinter import N
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# import sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------

# read the dataset into houses data
houses = pd.read_csv("./house_price_prediction/Houseprices.csv")

# ------------------------------------------------------------------------

# describe the dataset
print(houses.describe())

# Describe the sales price and the living area
print(houses['SalePrice'].describe())
print(houses['GrLivArea'].describe())

# ------------------------------------------------------------------------
# create 1D Array for linear regression

# set the x variable to house area
x = houses['GrLivArea'].values
# set the y variable to house price
y = houses['SalePrice'].values


# -----------------------------------------------------------------------

# plot the regression of the house prices against the living area
plt.scatter(x, y)
# plt.show()


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# QUESTION 1
# -----------------------------------------------------------------------
# Give the mathematical expression of the bias term theta(0) and the theta(1) of the
# regression line for predicting y from x.
# Calculate the regression parameters theta(0) and theta(1).
# Also calculate the regression parameters and the R2 variance score using the
# LinearRegression module in the sklearn.linear_model library . Check that your
# calculations of theta(0) and theta(1) are right.
# -----------------------------------------------------------------------
# QUESTION 2
# -----------------------------------------------------------------------
# Give the mathematical expression of the bias term #! $ and the slope #" $ of the
# regression line for predicting x from y.
# Calculate theta(0)` and theta(1)` .
# Also calculate the regression parameters and the R2 variance score using the
# LinearRegression module in the sklearn.linear_model library . Check that your
# calculations oftheta(0)` and theta(1)` are right.
# -----------------------------------------------------------------------
# REGRESSION
# -----------------------------------------------------------------------
# Analytical calcultion of regression coefficients
# -----------------------------------------------------------------------
x_squared = np.mean(np.multiply(x,x))
y_squared = np.mean(np.multiply(y,y))
xy = np.mean(np.multiply(x, y))
x_avg = np.mean(x)
y_avg = np.mean(y)

theta0 = (x_squared*y_avg-x_avg*xy)/(x_squared - x_avg*x_avg)
theta1 = (xy-x_avg*y_avg)/(x_squared - x_avg*x_avg)

theta0prime = (y_squared*x_avg-y_avg*xy)/(y_squared-y_avg*y_avg)
theta1prime = (xy-x_avg*y_avg)/(y_squared - y_avg*y_avg)

print("The regression parameters (theta0, theta1) : ", theta0, theta1)
print(" The regression parameters (theta1prime, theta0prime)", theta0prime,theta1prime)

# -----------------------------------------------------------------------
# Prediction of y using x  using linear regression in sklearn and 
# printing the associated weights and biases
# -----------------------------------------------------------------------

# call the sklearn model
regress  = LinearRegression(fit_intercept=True)

# predict y given x

regress.fit(x.reshape(-1,1),y)
y_ = regress.predict(x.reshape(-1,1))
print("theta0 , theta1 :", regress.intercept_, regress.coef_)
# predict y given x
regress.fit(y.reshape(-1,1),x)
x_ = regress.predict(y.reshape(-1,1))
print("Theta0prime, theta1prime ;", regress.intercept_,regress.coef_)


# PRINCIPAL COMPONENT ANALYSIS - Analytical method
t1 = y_squared - x_squared
t2 = t1**2
t3 = 4*xy**2

slope_ortho = (t1+np.sqrt(t2+t3))/(2*xy)

# PRINCIPAL COMPONENT ANALYSIS - Sklearn method

# Create and standardie 2D array for PCA and Orthogonal Regression
XY = houses[['GrLivArea','SalePrice']].values.astype(np.float64)
scaler = StandardScaler()
XY = scaler.fit_transform(XY)

# Apply PCA Analysis
pca = PCA(n_components=1)
pca.fit(XY)

print("Explained variance of PCA : ", pca.explained_variance_ratio_)
print("Componenets of PCA : ", pca.components_)


# calculate the slope of the pca components compared to the orthogonal regression regression
slope_pca = pca.components_[0,1]/pca.components_[0,0]
# -----------------------------------------------------------------------

# Plot the points on for the linear regression line
"""
Now plot all the results
"""
plt.figure  (figsize=(10, 8))
"""
Plot the input data in red
"""
plt.scatter (x, y, s=20, color='red')
"""
Plot the two Regression Lines (y vs x and x vs y) in blue and black
"""
plt.plot    (x  , y_ , color='blue'   ,linewidth=3, label='Regression y vs x')
plt.plot    (x_ , y  , color='black'  ,linewidth=3, label='Regression x vs y')
"""
Correct for the slopes for plotting, because of the initial scaling of the input data
"""
stddev      = np.sqrt(scaler.var_)
slope_ortho = slope_ortho * stddev[1]/stddev[0]
slope_pca   = slope_pca   * stddev[1]/stddev[0]
"""
Plot PCA and Orthogonal Regression lines, which should overlap (we should only see the green
line as it is the  second one to be plotted)
"""
plt.plot    (x,((np.mean(y) - slope_pca  *np.mean(x)) + slope_pca  *x), color='cyan'    ,linewidth=3, label= 'Orthogonal Regression')
plt.plot    (x,((np.mean(y) - slope_ortho*np.mean(x)) + slope_ortho*x), color='green'   ,linewidth=1, label= 'PCA')

plt.xlim(0.,6000.)
plt.ylim(0.,800000.)

plt.xlabel('Greater Living Area Above Ground')
plt.ylabel('House Sale Price')

plt.title ('Houses Sale Price: Compare Linear Regressions, Orthogonal Regression and PCA')

plt.legend()

plt.show()
plt.savefig('house_price_prediction/results_linear_regression_VS_PCA.png')

