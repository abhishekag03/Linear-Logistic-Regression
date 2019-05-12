import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
import statistics

train_RMSE = []
test_RMSE = []
num_iterations = 1000
learning_rate = 0.04
all_folds_train_RMSE = np.zeros((1000,5))
all_folds_test_RMSE = np.zeros((1000,5))

def linear_regression(training_set_x, test_set_x, training_set_y, test_set_y, fold_num) :
	num_training_samples = len(training_set_x)
	num_test_samples = len(test_set_x)
	parameters = np.zeros(14)
	error = 0
	sum = 0
	array_training_error = np.zeros(num_training_samples)
	array_test_error = np.zeros(num_test_samples)
	
	# array_training_error[2] = 2
	arr_training_error = []
	arr_test_error = []
	arr_iterations = []
	for k in range(num_iterations):
	
		for j in range(num_training_samples):
			diff_train = (np.dot(training_set_x[j], parameters) - training_set_y[j])
			array_training_error[j] = diff_train
		
		for j in range(num_test_samples):
			diff_test = (np.dot(test_set_x[j], parameters) - test_set_y[j])
			array_test_error[j] = diff_test

		#updating parameters
		for i in range(14):
			parameters[i] = parameters[i] - (learning_rate/num_training_samples)*(np.dot(array_training_error, training_set_x[:, i]))

		# print("print ",np.sum(array_training_error**2))
		error_training = math.sqrt(np.sum(array_training_error**2)/num_training_samples)
		error_test = math.sqrt(np.sum(array_test_error**2)/num_test_samples)
		# print ("error = ", error)
		arr_iterations.append(k)
		arr_training_error.append(error_training)
		all_folds_train_RMSE[k][fold_num] = error_training
		all_folds_test_RMSE[k][fold_num] = error_test

		arr_test_error.append(error_test)		
		
 	    
	plt.figure(1)
	plt.plot(arr_iterations, arr_training_error)
	plt.xlabel("Number of Iterations")
	plt.ylabel("Training RMSE of 5 folds")
	train_RMSE.append(arr_training_error[len(arr_training_error)-1])
	plt.figure(2)
	plt.plot(arr_iterations, arr_test_error)
	test_RMSE.append(arr_test_error[len(arr_test_error)-1])
	plt.xlabel("Number of Iterations")
	plt.ylabel("Test RMSE of 5 folds")
	# print(parameters)
	return parameters



if __name__ == '__main__':
	data = pd.read_csv(".\\boston_csv.csv")
# print(data.shape)

	y = data.values
	x = np.array(y)
	data_x = x[:, :-1]
	data_y = x[:, -1]
	data_x = np.insert(data_x, [0], [1], axis=1)
	
	for i in range(1, 14):
		data_x[:, i] = (data_x[:, i] - min(data_x[:, i]))/(max(data_x[:, i]) - min(data_x[:, i]))

	kfold = KFold(n_splits=5) 
	kfold.get_n_splits(x)
	fold_num = 0
	for train_index, test_index in kfold.split(data_x):
		# print (len(train_index), len(test_index))
		# print (train_index, test_index)
		x_train = data_x[train_index] 
		x_test = data_x[test_index]
		y_train = data_y[train_index]
		y_test = data_y[test_index]

		training_parameters = linear_regression(x_train, x_test, y_train, y_test, fold_num)
		fold_num +=1

		# print (np.dot(training_parameters, training_set_x[0]))
	for i in range(5):
		print ("Training RMSE of fold ", i+1, " is: ", train_RMSE[i])
		print ("Test RMSE of fold ", i+1, " is: ", test_RMSE[i])		

	print ("mean, standard deviation RMSE of training = (", statistics.mean(train_RMSE), " +- ", statistics.stdev(train_RMSE), 	")")
	print ("mean, standard deviation RMSE of test = (", statistics.mean(test_RMSE), " +- ", statistics.stdev(test_RMSE), ")")

	mean_train_RMSE = np.zeros(1000)
	mean_test_RMSE = np.zeros(1000)
	std_train_RMSE = np.zeros(1000)
	std_test_RMSE = np.zeros(1000)
	for i in range(1000):
		mean_train_RMSE[i] = statistics.mean(all_folds_train_RMSE[i])
		mean_test_RMSE[i] = statistics.mean(all_folds_test_RMSE[i])
		std_train_RMSE[i] = statistics.stdev(all_folds_train_RMSE[i])
		std_test_RMSE[i] = statistics.stdev(all_folds_test_RMSE[i])

	iterations_array =  np.arange(0,1000)
	# mean_RMSE = np.sum(all_folds_train_RMSE)/5.0/
	plt.figure(3)
	plt.bar(iterations_array, mean_train_RMSE, std_train_RMSE)
	plt.xlabel("Number of Iteartions")
	plt.ylabel("Mean and Standard Deviation of Train")
	plt.figure(4)	

	
	plt.bar(iterations_array[0::5], mean_test_RMSE[0::5], std_test_RMSE[0::5])

	# plt.bar(iterations_array, mean_test_RMSE, std_test_RMSE)
	plt.xlabel("Number of Iteartions")
	plt.ylabel("Mean and Standard Deviation of Test")

	plt.show()
