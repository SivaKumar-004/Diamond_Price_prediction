


# Diamond is one of the precious stones which are always in huge demand in the investment market. Diamonds are also used in many industrial applications like cutting but it is mostly used as a gemstone. The actual price of a diamond however is determined by a gemologist after examining its various features such as its carat, cut, color, and clarity. Dimensions of a diamond is also a very important parameter to determine its worth. Nearly, 142 million carats of diamonds were produced worldwide in 2019 alone. This makes it very important to come up with some smart technique to estimate its worth.


# A diamond distributor decided to put almost 2000 diamonds for auction. A jewellery company is interested in making a bid to purchase these diamonds in order to expand their business. As a data scientist, your job is to build a prediction model to predict the price of diamonds so that your company knows how much it should bid.


# 1. Explore the diamond dataset by creating the following plots:
# 4. Build a linear regression model by selecting the most relevant features to predict the price of diamonds.
# 5. Evaluate the linear regression model by calculating the parameters such as coefficient of determination, MAE, MSE, RMSE, mean of residuals and by checking for homoscedasticity.


# #### 1. Import Modules and Load Dataset
# Link to the dataset: https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/diamonds.csv



# Import the required modules and load the dataset.
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/diamonds.csv')
df.head()




# Get the information on DataFrame.
df.info()




df.isnull().sum()




df = df.drop(columns='Unnamed: 0', axis=1)
df.head()



# #### 2. Exploratory Data Analysis



import matplotlib.pyplot as plt
import seaborn as sns




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Boxplot for 'cut' vs 'price'",fontsize = 20, color='teal')
sns.boxplot(x='cut', y='price', data=df)
plt.xlabel("Cut", color='teal')
plt.ylabel("Price", color='teal')
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Boxplot for 'color' vs 'price'",fontsize = 20, color='teal')
sns.boxplot(x='color', y='price', data=df)
plt.xlabel("Color", color='teal')
plt.ylabel("Price", color='teal')
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Boxplot for 'clarity' vs 'price'",fontsize = 20, color='teal')
sns.boxplot(x='clarity', y='price', data=df)
plt.xlabel("Clarity", color='teal')
plt.ylabel("Price", color='teal')
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'carat' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='carat', y='price', data=df, color='navy')
plt.axvline(df['carat'].mean(), color='r', label=f"Mean: {df['carat'].mean()}")
plt.xlabel("Carat", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'depth' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='depth', y='price', data=df, color='navy')
plt.axvline(df['depth'].mean(), color='r', label=f"Mean: {df['depth'].mean()}")
plt.xlabel("Depth", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'table' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='table', y='price', data=df, color='navy')
plt.axvline(df['table'].mean(), color='r', label=f"Mean: {df['table'].mean()}")
plt.xlabel("Table", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'x' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='x', y='price', data=df, color='navy')
plt.axvline(df['x'].mean(), color='r', label=f"Mean: {df['x'].mean()}")
plt.xlabel("X", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'y' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='y', y='price', data=df, color='navy')
plt.axvline(df['y'].mean(), color='r', label=f"Mean: {df['y'].mean()}")
plt.xlabel("Y", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter Plot for 'z' and 'price'",fontsize = 20, color='teal')
sns.scatterplot(x='z', y='price', data=df, color='navy')
plt.axvline(df['z'].mean(), color='r', label=f"Mean: {df['z'].mean()}")
plt.xlabel("Z", color='teal')
plt.ylabel("Price", color='teal')
plt.legend()
plt.grid()
plt.show()





plt.figure(figsize=(20,7), dpi=96)
plt.style.use('dark_background')
plt.title("Normal Distribution Curve for 'price'",fontsize = 20, color='teal')
sns.distplot(x=df['price'], bins='sturges', hist=False, color='navy')
plt.axvline(df['price'].mean(), color='r', label=f"Mean: {df['price'].mean():.2f}")
plt.xlabel("Price", color='teal')
plt.ylabel("Density", color='teal')
plt.legend()
plt.grid()
plt.show()
def pro_den(series,mean,std):
  coeff = 1/(std*np.sqrt(2*np.pi))
  power_e = np.exp(-(series-mean)**2/(2*std**2))
  pro = coeff*power_e
  return pro

print('~'*150)
plt.figure(figsize=(20,7), dpi=96)
plt.style.use('dark_background')
plt.title("Normal Distribution Curve using plt.scatter() for 'price'",fontsize = 20, color='teal')
plt.scatter(df['price'], pro_den(df['price'], df['price'].mean(), df['price'].std()), color='navy')
plt.axvline(df['price'].mean(), color='r', label=f"Mean: {df['price'].mean():.2f}")
plt.xlabel("Price", color='teal')
plt.ylabel("Density", color='teal')
plt.legend()
plt.grid()
plt.show()




# The dataset contains certain features that are categorical.  To convert these features into numerical ones, use `replace()` function of the DataFrame.
# `df["column1"].replace({"a": 1, "b": 0}, inplace=True)` $\Rightarrow$ replaces all the `'a'` values with `1` and `'b'` values with `0` for feature `column1`. Use `inplace` boolean argument to to make changes in the DataFrame permanently.



df.head()




df['cut'].replace({'Fair':1, 'Good':2, 'Very Good':3, 'Premium':4, 'Ideal':5}, inplace=True)




df['color'].replace({'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7}, inplace=True)




df['clarity'].replace({'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}, inplace=True)




df.head()



# #### 4. Model Training
# Build a multiple linear regression model  using all the features of the dataset. Also, evaluate the model by calculating $R^2$, MSE, RMSE, and MAE values.



feature = list(df.columns[:])
feature.remove('price')
feature




# Build multiple linear regression model using all the features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = df[feature]
y = df['price']
# Split the DataFrame into the train and test sets such that test set has 33% of the values.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# Build linear regression model using the 'sklearn.linear_model' module.
model = LinearRegression()
model.fit(X_train, y_train)
print("Constant".ljust(15, " "), f"\t{model.intercept_:.6f}")
for item in list(zip(X_train.columns.values, model.coef_)):
  print(f"{item[0]}".ljust(15, " "), f"\t{item[1]:.6f}")




# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Train Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_train, y_train_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.3f}")

print(f"Test Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_test, y_test_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.3f}")


# **Q:** What is the $R^2$ (R-squared) value for this model?
# **A:** The $R^2$ (R-squared) value for this model is `0.907`.





# Heatmap to pinpoint the columns in the 'df' DataFrame exhibiting high correlation
plt.figure(figsize=(20,7))
sns.heatmap(df.corr(), annot=True)
plt.show()


# **Q:** Is there multicollinearity in the dataset?
# **A:** `Yes`, there are muticollinearity in the dataset.




df = df.drop(columns=['x', 'y', 'z'])
df.head()




# Again build a linear regression model using the remaining features
new_feature = list(df.columns[:])
new_feature.remove('price')
x_train_new = X_train.iloc[:,:-3]
x_test_new = X_test.iloc[:,:-3]
# Build linear regression model using the 'sklearn.linear_model' module.
model_2 = LinearRegression()
model_2.fit(x_train_new, y_train)
print("Constant".ljust(15, " "), f"\t{model_2.intercept_:.6f}")
for item in list(zip(x_train_new.columns.values, model_2.coef_)):
  print(f"{item[0]}".ljust(15, " "), f"\t{item[1]:.6f}")




# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
y_train_new_pred = model_2.predict(x_train_new)
y_test_new_pred = model_2.predict(x_test_new)

print(f"Train Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_train, y_train_new_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_new_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_new_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_new_pred):.3f}")

print(f"\nTest Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_test, y_test_new_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_new_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_new_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_new_pred):.3f}")





import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

x_train_sm = sm.add_constant(x_train_new)
# Create a dataframe that will contain the names of the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_sm.columns
vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.values.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif












# Again build a linear regression model using the features whose VIF values are less than 10

# Build linear regression model using the 'sklearn.linear_model' module.






# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.










# Create a histogram for the errors obtained in the predicted values for the train set.
error_train = y_train - y_train_new_pred
plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Histogram for Errors in Predicted Values for the 'Train set'.",fontsize = 20, color='teal')
plt.hist(error_train, bins='sturges', color='navy')
plt.axvline(error_train.mean(), color='r', label=f"Mean: {error_train.mean():.2f}")
plt.xlabel("Error", color='teal')
plt.ylabel("Count", color='teal')
plt.legend()
plt.grid()
plt.show()




error_test = y_test - y_test_new_pred
plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Histogram for Errors in Predicted Values for the 'Test set'.",fontsize = 20, color='teal')
plt.hist(error_test, bins='sturges', color='navy')
plt.axvline(error_test.mean(), color='r', label=f"Mean: {error_test.mean():.2f}")
plt.xlabel("Error", color='teal')
plt.ylabel("Count", color='teal')
plt.legend()
plt.grid()
plt.show()


# **Q:** Is the mean of errors equal to 0 for train set?
# **A:** `Yes` the mean of errors equal to 0 for train set.





# Create a scatter plot between the errors and the dependent variable for the train set.
plt.figure(figsize=(20,7))
plt.style.use('dark_background')
plt.title("Scatter plot between the errors and dependent variable for the 'train set'",fontsize = 20, color='teal')
sns.scatterplot(y_train, error_train, color='navy')
plt.axhline(error_train.mean(), color='r', label=f"Mean of errors: {error_train.mean():.2f}")
plt.xlabel("Price", color='teal')
plt.ylabel("Error", color='teal')
plt.legend()
plt.grid()
plt.show()




#   <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/2_share_button.png' width=500>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/3_copy_link.png' width=500>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/4_copy_link_confirmation.png' width=500>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/5_student_dashboard.png' width=800>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/6_my_projects.png' width=800>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/7_view_project.png' width=800>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/8_submit_project.png' width=800>
#    <img src='https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/images/project-share-images/9_enter_project_url.png' width=800>

