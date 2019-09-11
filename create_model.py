import tensorflow as tf
import cv2
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import json
from train_model import Train_model
from tensorflow.keras.datasets import boston_housing

def regression_model(no_of_hidden_layers,hidden_unit_size,activation_function,optimizer_func):
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(13,input_dim=13,activation=activation_function))
	for i in range(no_of_hidden_layers):
		model.add(tf.keras.layers.Dense(hidden_unit_size,activation=activation_function))
	model.add(tf.keras.layers.Dense(1,activation='tanh'))
	model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
	return model

def classsification_model(no_of_hidden_layers,hidden_unit_size,activation_function,eval_metrics,optimizer_func):
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten())
	for i in range(no_of_hidden_layers):
		model.add(tf.keras.layers.Dense(hidden_unit_size,activation=activation_function))
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=eval_metrics)
	return model

def return_model(no_of_hidden_layers,hidden_unit_size,activation_function,category,eval_metrics,optimizer_func):
	if category==1:
		model=classsification_model(no_of_hidden_layers,hidden_unit_size,activation_function,optimizer_func)
	elif category==2:
		model=regression_model(no_of_hidden_layers,hidden_unit_size,activation_function,eval_metrics,optimizer_func)
	return model
	
		
	
def new_model_create():
	print('CATEGORIES: \n1\t Classification\n2\t Regression')
	category=int(input('Enter the category'))
	#input_size=int(input('Enter the input dimension'))
	hidden_units=int(input('Enter the no. of hidden layers:'))
	hidden_unit_size=int(input('Enter the no. of neurons in hidden layer:'))
	print('activation functions: \n 1\t relu\n 2\tsoftmax\n 3\t tanh\n 4\tsigmoid\n')
	activation_function=int(input('Enter the activation function number'))
	functions={1:'relu',2:'softmax',3:'tanh',4:'sigmoid'}
	metrics_dict={1:'accuracy',2:'binary_accuracy',3:'categorical_accuracy'}
	if category==1:
		print('Enter the metrics for evaluation (enter space separated values):\n 1. accuracy\n 2. binary_accuracy\n 3. categorical_accuracy')
		metrics=[int(x) for x in input().split(' ')]
		eval_metrics=[]
		for i in metrics:
			eval_metrics.append(metrics_dict[i])
	else:
		metrics=['mae']
	print('Enter the optimizer:\n 1. Adagrad\n 2. Adadelta\n 3. Adam\n 4.Nadam')
	optimizer=int(input())
	optimizer_dict={1:'Adagrad',2:'Adadelta',3:'Adam',4:'Nadam'}
	model=return_model(hidden_units,hidden_unit_size,functions[activation_function],category,eval_metrics,optimizer_dict[optimizer])
	config={'no_of_hidden_layers':hidden_units,'hidden_unit_size':hidden_unit_size,'activation_function':functions[activation_function],'category':category,'metrics':eval_metrics,'optimizer':optimizer_dict[optimizer]}
	with open('config.json','w') as json_file:
		json.dump(config,json_file)	
	#Train_model(model,3,'config.json')
	model.save('ann.h5')
	#load_model('ann.h5')
	#print('model loaded successfully')
	
if __name__=='__main__':
	new_model_create()
