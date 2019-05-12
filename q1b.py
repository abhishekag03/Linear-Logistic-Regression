import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
import statistics
from sklearn.model_selection import  GridSearchCV
# from sklearn.model_selection import SVC
from sklearn import linear_model



train_RMSE = []
num_graph = 1
def regularized_gradient_descent(norm, training_set_x, test_set_x, training_set_y, test_set_y):
	
	learning_rate = 0.04
	alpha_dict = {'alpha' : [100, 50, 10, 4, 3.5, 3, 2.5, 2, 1, 0.1, 0.005, 0.003]}
	num_training_samples = len(training_set_x)
	num_test_samples = len(test_set_x)
	parameters = np.zeros(14)
	error = 0
	sum = 0
	array_training_error = np.zeros(num_training_samples)
	array_test_error = np.zeros(num_test_samples)
	num_iterations = 1000

	arr_test_error = []
	arr_training_error = []
	arr_iterations = []

	if (norm == 'L2'):
		print ("Running L2")
		output = GridSearchCV(linear_model.Ridge(), alpha_dict, 'neg_mean_squared_error', cv = 5)
	else:
		print ("Running L1")		
		output = GridSearchCV(linear_model.Lasso(), alpha_dict, 'neg_mean_squared_error', cv = 5)

	output.fit(training_set_x, training_set_y)
	# print(output.best_param)
	# print(output.best_estimator_.get("alpha"))
	print("Params = ",output.best_params_.get("alpha"))

	best_estimator = output.best_params_.get("alpha")
	for k in range(num_iterations):
	
		for j in range(num_training_samples):
			diff_train = (np.dot(training_set_x[j], parameters) - training_set_y[j])
			array_training_error[j] = diff_train

		for j in range(num_test_samples):
			diff_test = (np.dot(test_set_x[j], parameters) - test_set_y[j])
			array_test_error[j] = diff_test

		#updating parameters
		for i in range(14):
			if (i == 0):
				parameters[i] = parameters[i] - (learning_rate/num_training_samples)*(np.dot(array_training_error, training_set_x[:, i]))
				continue
			if (norm == 'L2'):
				parameters[i] = parameters[i] - (learning_rate/num_training_samples)*((np.dot(array_training_error, training_set_x[:, i])) + best_estimator*parameters[i])
			else:
				parameters[i] = parameters[i] - (learning_rate/num_training_samples)*((np.dot(array_training_error, training_set_x[:, i])) + best_estimator)

		error_training = math.sqrt(np.sum(array_training_error**2)/num_training_samples)
		# print ("error = ", error)
		error_test = math.sqrt(np.sum(array_test_error**2)/num_test_samples)

		arr_iterations.append(k+1)
		arr_training_error.append(error_training)
		arr_test_error.append(error_test)

	    
	plt.figure(num_graph)
	plt.plot(arr_iterations, arr_training_error)
	train_RMSE.append(arr_training_error[len(arr_training_error)-1])
	print ("Training RMSE after ", norm, " regularization is: ",arr_training_error[len(arr_training_error)-1])
	print ("Test RMSE after ", norm, " regularization is: ", arr_test_error[len(arr_test_error)-1])
	return parameters

	
if __name__ == '__main__':
	
	data = pd.read_csv(".\\boston_csv.csv")

	# print(data.shape)
	y = data.values
	x = np.array(y)
	data_set_x = x[:, :-1]
	data_set_y = x[:, -1]
	data_set_x = np.insert(data_set_x, [0], [1], axis=1)




	for i in range(1, 14):
		data_set_x[:, i] = (data_set_x[:, i] - min(data_set_x[:, i]))/(max(data_set_x[:, i]) - min(data_set_x[:, i]))

	training_validation_x = data_set_x[103:506]
	training_validation_y = data_set_y[103:506]
	test_x = data_set_x[0:103]
	test_y = data_set_y[0:103]
	test_errors = list()
	regularized_gradient_descent('L2', training_validation_x, test_x, training_validation_y, test_y)
	num_graph+=1
	regularized_gradient_descent('L1', training_validation_x, test_x, training_validation_y, test_y)

	plt.show()
	# output = GridSearchCV(linear_model.Ridge(), alpha_dict, cv = 5)

	# output.fit(training_validation_x, training_validation_y)
	# test_errors.append(output.score(test_x, test_y))

	# print (output.best_params_)


	# # refit()
	# predict2 = output.predict(training_validation_x)
	# prediction = output.predict(test_x)
	# check = output.score(test_x, test_y)
	# # print (test_validation_y)
	# print(output.best_estimator_)
	# # print (prediction)
	# error = 0
	# print("te = ", test_errors)
	# for i in range(len(training_validation_y)):
	# 	error = error + (training_validation_y[i] - predict2[i])**2

	# rmse = math.sqrt(error/len(training_validation_x))
	# print ("ab = ", rmse)
	# rmse = 0;
	# error = 0
	# for i in range(len(test_y)):
	# 	error = error + (test_y[i] - prediction[i])**2

	# rmse = math.sqrt(error/len(test_y))
	# print (check)

	# print (training_validation_x)


