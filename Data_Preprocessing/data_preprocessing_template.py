# Data Preprocessing Template

"""
using Import "as" means that you can refer to the library later on with 
an abreviationi.e np.

matplotlib - for visualising stuff in python

pandas - Best library for importing and managing datasets (like .csv)


"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
"""
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# =============================================================================
# #Dataset imports the data and dataset.iloc creates the matrix (like an array)
# #iloc - reads the table and : means import all the lines
# #, = left of the comma = the lines (rows) of the dataset
# #, = right of the dataset = all the columns minus the last one -1
# #--
# #
# #in the case of y we just take the last column
# =============================================================================
"""
#taking care of missing data
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0) #Define the Imputer CLASS
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) #Add the mean data to the missing columns using Transform Method
# =============================================================================
# Basically gonna use the mean of any ohter values ina  given row
# 
# "from" imports the library- then "import" the library

# #fix the imputer to the Matrix X
# #take all the rows in X and take coloumns between 1 and 2 
# #we use 1:3 because in python the last row is ignored
# =============================================================================
"""
# Encoding the categorical Data
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0]) #This encodes the data in the X column
#Mask the value of the newly defined encoded integers 
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
#This is the dependent variable column 3 so the ML knows there is no value diff between them
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  
"""
# Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split #library to split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# =============================================================================
# X_train is the training set for the X table
# X_test is the test set of the matrix of features 
# Y_train is the training part of the data associated to the X matrix
# _test is teh test part of the dependent varible vector associated to X_test
# all variables assigned at the same time with = 
# =============================================================================
"""
# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #initiate the class
X_train = sc_X.fit_transform(X_train) #recompute X-train as we want to scale this
X_test = sc_X.transform(X_test) #Do the same for the Test Set for X

# =============================================================================
# sc_y = StandardScaler() #Also scale the dummy variables for countries
# y_train = sc_y.fit_transform(y_train) 
# =============================================================================













