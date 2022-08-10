'''
Regression model binary classification using step function iris dataset
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random 

def load_data():
	'''
	Replace output variable with 0,1,2. Remove Viriginica and scale data between 0 and 1
	'''
	df = pd.read_csv('iris.csv')
	df = df.replace({'Setosa':0,'Versicolor':1,'Virginica':2})
	df = df[(df.variety == 0) | (df.variety == 1)]
	scaler = MinMaxScaler()
	df.iloc[:,0:4] = scaler.fit_transform(df.iloc[:,0:4])
	return df

def train_test_data_split(data_frame,expnum):
	'''
	Split the training and test data 80/20 where the random state is based on experiment number
	'''
	X = data_frame.iloc[:,0:4].to_numpy()
	Y = data_frame.iloc[:,4].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.8,random_state=expnum)
	return xtrain,xtest,ytrain,ytest

def linear_reg_model(xtrain,xtest,ytrain,ytest):
	'''
	Initialize a linear regression model, fit the training data to the model and predict the 
	flower type using the fitted model and the test data. Then pass the predicted values into 
	heaviside function to ensure values of 0 or 1
	'''
	regr = linear_model.LinearRegression()
	regr.fit(xtrain,ytrain)
	ypred = regr.predict(xtest)
	stepypred = np.heaviside(ypred,1)
	return stepypred

def thirty_experinments(data_frame):
	'''
	Run 30 experiments for the linear regression model
	'''
	acc_lst = np.empty(30)
	for exp in range(30):
		xtrain,xtest,ytrain,ytest = train_test_data_split(data_frame,exp)
		ypred = linear_reg_model(xtrain,xtest,ytrain,ytest)
		acc_lst[exp] = accuracy_score(ypred,ytest)
	return acc_lst

def main():
	df = load_data()
	experinment_num = 0
	xtrain,xtest,ytrain,ytest = train_test_data_split(df,experinment_num)
	ypred = linear_reg_model(xtrain,xtest,ytrain,ytest)
	print('\nOne experinment results \n')
	print(ypred,' predicted y values')
	print(ytest,' actual y values')
	print(accuracy_score(ypred,ytest),' classification accuracy')
	acc_lst = thirty_experinments(df)
	print(acc_lst, 'accuracy for 30 differnt splits')
	print('\naccuracy score mean',acc_lst.mean())

if __name__ == "__main__":
	main()