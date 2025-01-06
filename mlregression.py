link = "https://raw.githubusercontent.com/swapnilsaurav/MachineLearning/refs/heads/master/2_Marks_Data.csv"
# Prediction of marks based on hours of study
import  pandas as pd
data = pd.read_csv(link)
print(data)

X = data.iloc[:, :1].values
y = data.iloc[:, 1].values
print("1.  X : ",X)
print("1.  y : ",y)

# EDA - to understand the data
# import matplotlib.pyplot as plt
# plt.scatter(data['Hours'], data['Marks'])
# plt.show()

# Dividing into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1)
print("X_train: \n",X_train)
print("X_test: \n",X_test)
print("y_train: \n",y_train)
print("y_test: \n",y_test)


# Run the regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# equation of the best fit line;
m = regressor.coef_
c = regressor.intercept_
print("Coefficient/Slope : ",m)
print("Intercept/Constant : ",c)
print(f"Equation of the best fit line: {m}x + {c} ")

# to see the best fit line
import matplotlib.pyplot as plt
# taking entire data
# plt.scatter(X_train, y_train, color="blue")
# # regression line on the training data
# plt.plot(X_train, m*X_train+c, color="red")
# plt.show()

y_pred = regressor.predict(X_test)
out_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print("Predicted v Actual - Linear Regression Model \n",out_df)


# Model Evaluation:
# Regression metric: Mean Absolute Error (MAE), Mean Squared Error (MSE)
#     Root Mean Squared Error (RMSE), R Square value
#     (compares how well we have done wrt to average)
from sklearn import metrics
print("============  VALIDATION ERROR / TEST ERROR  =============")
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("MAE = ",mae)
print("MSE = ",mse)
print("RMSE = ",mse**0.5)
print("R Square Score = ",r2)
