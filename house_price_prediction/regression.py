

# import packages for numerical manipulations and visulatizations
import pandas as pd # for manipulating tabular data
import matplotlib.pyplot as plt # for visualisation
import seaborn as sns # for user friendly visualisation
import numpy as np # for numerical python functionality

# import packages for linear regression
from sklearn.linear_model import LinearRegression # implementation of linear regression
from sklearn.model_selection import train_test_split # for creating a train and test set
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # for evaluating our model

# -----------------------------------------------------------------
# import data
# -----------------------------------------------------------------

houses = pd.read_csv('./house_price_prediction/Houseprices.csv')

# -----------------------------------------------------------------
# Exploratory Data Analysis
# -----------------------------------------------------------------

# view the first 5 rows of the dataset along with the columns names
print(houses.head())

# number of rows and columns in houses
print(houses.shape)

# column names 
print(houses.columns)

# info provides an overview of the data-type and number of real data-points present for each column
print(houses.info())


# Consider only numerical data types
# for now we are only going to consider numerical data-types 
houses_num = houses.select_dtypes(include = 'number')
print(houses_num.info())

#check for missig values
missing = houses.isna().sum().sort_values(ascending=False)
print(missing.head(20))

# -----------------------------------------------------------------
# Dealing with missing data
# -----------------------------------------------------------------

# how many missing values do we have from purely numerical features
houses_num.isna().sum().sort_values(ascending=False).head()


# -----------------------------------------------------------------
# Cleaning data
# -----------------------------------------------------------------

# in the interest of time we will simply drop the LotFrontage column
houses_num = houses_num.drop('LotFrontage', axis=1)

# we will then drop all rows containing the remaining missing values 
houses_num = houses_num.dropna()

print(houses_num.isna().sum())

# -----------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------

# distribution of target variable
sns.displot(houses_num['SalePrice'], kde=True)
plt.grid(alpha=0.5)

# visualise the relationship between different features and the target variable
sns.pairplot(houses_num, x_vars=houses_num.columns.drop('SalePrice')[:8], y_vars=['SalePrice'], markers='.',
             plot_kws={'x_jitter': 0.1, 'y_jitter': 0.1, 'scatter_kws': {'alpha': 0.2}},
             kind='reg')
# visualise the relationship between different features and the target variable
sns.pairplot(houses_num, x_vars=houses_num.columns.drop('SalePrice')[8:16], y_vars=['SalePrice'], markers='.',
             plot_kws={'x_jitter': 0.1, 'y_jitter': 0.1, 'scatter_kws': {'alpha': 0.2}},
             kind='reg')

# visualise the relationship between different features and the target variable
sns.pairplot(houses_num, x_vars=houses_num.columns.drop('SalePrice')[16:24], y_vars=['SalePrice'], markers='.',
             plot_kws={'x_jitter': 0.1, 'y_jitter': 0.1, 'scatter_kws': {'alpha': 0.2}},
             kind='reg')


# visualise the relationship between different features and the target variable
sns.pairplot(houses_num, x_vars=houses_num.columns.drop('SalePrice')[24:32], y_vars=['SalePrice'], markers='.',
             plot_kws={'x_jitter': 0.1, 'y_jitter': 0.1, 'scatter_kws': {'alpha': 0.2}},
             kind='reg')

# visualise the relationship between different features and the target variable
sns.pairplot(houses_num, x_vars=houses_num.columns.drop('SalePrice')[32:], y_vars=['SalePrice'], markers='.',
             plot_kws={'x_jitter': 0.1, 'y_jitter': 0.1, 'scatter_kws': {'alpha': 0.2}},
             kind='reg')

# visualise correlation coefficients
plt.figure(figsize=(11, 11))
sns.heatmap(houses_num.corr(), square=True, cmap='RdBu_r', vmin=-1, vmax=1)


# -----------------------------------------------------------------
# Feature Selection
# -----------------------------------------------------------------
# correlation with the target variable
print(houses_num.corr()['SalePrice'].abs().sort_values(ascending=False))

# select the 9 most highly correlated features with the target variable to be our features
X = houses_num[houses_num.corr()['SalePrice'].abs().sort_values(ascending=False)[1:10].index.to_list()]

# select our target variable
y = houses_num["SalePrice"]

# -----------------------------------------------------------------
# Train Test Split
# -----------------------------------------------------------------

# perform train test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Check the shapes
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# -----------------------------------------------------------------
# CREATE THE MODEL
# -----------------------------------------------------------------

# instantiate the model with the class object
model = LinearRegression()

# fit model to training set
model.fit(X_train,y_train)

# -----------------------------------------------------------------
# EVALUATE THE MODEL
# -----------------------------------------------------------------

# predict the data using test data
preds_test = model.predict(X_test)

# evaluate how good these predictions are using mae and rmse
mae = mean_absolute_error(y_test, preds_test)
rmse = mean_squared_error(y_test, preds_test, squared=False)

print('MAE:', mae)
print('RMSE:', rmse)

# visual evaluation
# evaluation can also be done visually by plotting predictions vs the true target
# across the entire dataset
preds = model.predict(X)

plt.figure(figsize=(8,5))
plt.scatter(y, preds, alpha=0.7, linewidths=0.5, edgecolors='black')
plt.plot(y, y, color='darkred', alpha=0.5)
plt.xlabel('Real Sale Price')
plt.ylabel('Predicted Sale Price')
plt.grid(alpha=0.5)

# -----------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------

# instantiate model class object
model = LinearRegression()

# when using regression for inference you should fit your model to the entire dataset
model.fit(X, y)

# make predictions across the entire dataset
preds = model.predict(X)

# when using regression for inference it is more common to use the R-squared metric for evaluation 
# this is also typically done on the whole dataset (rather than a separate test set)
# it is a measure of how much variance in the target variable our model is able to explain 
r2 = r2_score(y, preds)
print("The R2 score of the dataset is : ",r2)

# Show the model features and parameters 
# lets have a look at our models parameters or coefficients
params = pd.DataFrame({'Features': X.columns, 'Coefficients': model.coef_})
print(params)


# -----------------------------------------------------------------
# MITIGATE MULTICOLLINARITY
# -----------------------------------------------------------------


# Visualize the correaltions int the parameters
plt.figure(figsize=(9, 9))
sns.heatmap(X_train.corr(), square=True, cmap='RdBu_r', vmin=-1, vmax=1, annot=True)
plt.savefig('./house_price_prediction/')