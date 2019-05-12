
from sklearn.datasets import fetch_mldata as sklearn_data
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def modify_y(current_class, y_values):
	new_y = []
	for i in range(len(y_values)):
		if (y_values[i] == current_class):
			new_y.append(1)
		else:
			new_y.append(0)

	return new_y




if __name__ == '__main__':

	mnist_dataset = sklearn_data('MNIST original')


	train_x, test_x, train_y, test_y = train_test_split(mnist_dataset.data, mnist_dataset.target, test_size=1/7.0, random_state=0)
	# train_x = train_x[:3000]
	# train_y = train_y[:3000]
	# test_x = test_x[:1000]
	# test_y = test_y[:1000]

	# print(np.shape(train_y))
	num_classes = 10
	l1_model_score =[]
	l2_model_score =[]
	l1_training_models=[]
	l2_training_models=[]
	# array_training = np.empty((1, len(train_x)), int)
	# array_test = np.empty((1,len(test_x)), int)
	for j in range(2):

		for i in range(num_classes):
			modified_train_y = modify_y(i, train_y)
			modified_test_y = modify_y(i, test_y)
			# l1_model = LogisticRegression(penalty="l1")
			if (j == 0):
				l1_model = LogisticRegression(penalty="l2", solver='lbfgs')
			else:
				l1_model = LogisticRegression(penalty="l1")

		# l2_model = LogisticRegression(penalty="l2", solver='lbfgs')

			l1_model.fit(train_x, modified_train_y)
			l1_training_models.append(l1_model)
			l1_coefficients = l1_model.coef_
			arr_train = np.dot(train_x, np.transpose(l1_coefficients))
			arr_train = 1.0/(1 + np.exp(-1*arr_train))
			# # array_training.append(arr_train)
			# l2_model = LogisticRegression(penalty="l2", solver='lbfgs')
			# l2_model.fit(train_x, modified_train_y)

			if (i == 0):
				array_training = arr_train
			else:
				array_training = np.append(array_training, arr_train, axis = 1)
			# print ("shape = ", np.shape(array_training))
			arr_test = np.dot(test_x, np.transpose(l1_coefficients))
			arr_test = 1.0/(1 + np.exp(-1*arr_test))
			if (i == 0):
				array_test = arr_test
			else:
				array_test = np.append(array_test, arr_test, axis = 1)


			# print ("sh =", array_test.shape)
			# np.append(array_training, arr, axis=1)
			# print (np.shape(array))
			# l2_model = LogisticRegression(penalty="l2")
			# l2_model.fit(train_x, modified_train_y)
			# l2_training_models.append(l2_model)

			# l1_model_score.append(l1_model.score(test_x, modified_test_y))
			# l2_model_score.append(l2_model.score(test_x, modified_test_y))

		# print("check = ",np.asarray(array_training).transpose().shape)
		# print(array_training[0])
		# print (a)
		actual_count = [0]*10
		predicted_count = [0]*10
		a_actual_count = [0]*10
		a_predicted_count = [0]*10
		test_predictions = len(test_x)
		training_predictions = len(train_x)
		correct_test_predictions = 0
		correct_training_predictions = 0

		for i in range(len(test_x)):
			max_prob_model = np.argmax(array_test[i])
			actual_count[int(test_y[i])]+=1
			# print (max_prob_model, test_y[i])
			if (max_prob_model == test_y[i]):
				predicted_count[int(test_y[i])]+=1
				correct_test_predictions+=1

		for i in range(len(train_x)):
			max_prob_model = np.argmax(array_training[i])
			a_actual_count[int(train_y[i])]+=1
			# print (max_prob_model, train_y[i])
			if (max_prob_model == train_y[i]):
				a_predicted_count[int(train_y[i])]+=1
				correct_training_predictions+=1


		if (j==0):
			print ("Printing results for L1")
		else:
			print ("Printing results for L2")

		print(actual_count, predicted_count)
		print(a_actual_count, a_predicted_count)
		print ((np.array(a_predicted_count)/np.array(a_actual_count))*100)
		print ((np.array(predicted_count)/np.array(actual_count))*100)
		

		
		print ("Train Accuracy = ", correct_training_predictions/training_predictions)
		print ("Test Accuracy = ", correct_test_predictions/test_predictions)

	# print (l1_model_score)
	# print (l2_model_score)

