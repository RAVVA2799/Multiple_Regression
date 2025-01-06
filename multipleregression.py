link="https://raw.githubusercontent.com/swapnilsaurav/MachineLearning/refs/heads/master/3_Startups.csv"
import pandas as pd
data = pd.read_csv(link)
print(data)

'''
Features (input): R&D Spend,Administration, Marketing Spend,State
Target (output): Profit 
Objective: to predict how much a start-up would generate 
    based on R&D Spend,Administration,  Marketing Spend,State

Dataset: We have the data in the above csv file
    We divide into X (input cols) and y (target col)
'''
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values
print("1.  X : ",X)
print("1.  y : ",y)
'''
Preprocessing:
    1. Handling of categorical data: State
    2. Dividing into Train and Test data
'''
# 1. handling of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lc_x = LabelEncoder()
X[:,3] = lc_x.fit_transform(X[:,3])
print("3.  X after Label Encoder: \n",X)

#column tranform
from sklearn.compose import ColumnTransformer
transfomer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = transfomer.fit_transform(X)
# drop any one column
X = X[:,1:]
print("4.  X after Column Transform and OneHotEncoder: \n",X)


# Dividing into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)
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
# Florida  X1, New York 2, R&D X3, Admin X4, Mkt  X5
'''
Coefficient/Slope :  [ 5.82738646e+02  2.72794662e+02  7.74342081e-01 -9.44369585e-03
  2.89183133e-02]
Intercept/Constant :  49549.70730374818
'''

# to see the best fit line
import matplotlib.pyplot as plt
# Profit v R&D
plt.scatter(data['R&D Spend'], data['Profit'], color="blue")
plt.show()
# Profit v Admin
plt.scatter(data['Administration'], data['Profit'], color="blue")
plt.show()
# Profit v Mkt
plt.scatter(data['Marketing Spend'], data['Profit'], color="blue")
plt.show()
