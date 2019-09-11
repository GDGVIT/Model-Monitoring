import tensorflow as tf
import cv2
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import json

def return_model(input_size,no_of_hidden_layers,hidden_unit_size,activation_function,category,classes,eval_metrics,optimizer_func):
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(input_size,activation=activation_function))
	for i in range(no_of_hidden_layers):
		model.add(tf.keras.layers.Dense(input_size,activation=activation_function))
	if category==1:
		model.add(tf.keras.layers.Dense(classes,activation=tf.nn.softmax))
		model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=eval_metrics)
	else:
		model.add(tf.keras.layers.Dense(classes,activation='tanh'))
		model.compile(optimizer=optimizer_func,loss='mean_squared_error',metrics=eval_metrics)
	
	return model
	
		
	
if __name__=='__main__':
	print('CATEGORIES: \n1\t Classification\n2\t Regression')
	category=int(input('Enter the category'))
	input_size=int(input('Enter the input dimension'))
	hidden_units=int(input('Enter the no. of hidden layers:'))
	hidden_unit_size=int(input('Enter the no. of neurons in hidden layer:'))
	print('activation functions: \n 1\t relu\n 2\tsoftmax\n 3\t tanh\n 4\tsigmoid\n')
	activation_function=int(input('Enter the activation function number'))
	functions={1:'relu',2:'softmax',3:'tanh',4:'sigmoid'}
	metrics_dict={1:'accuracy',2:'binary_accuracy',3:'categorical_accuracy'}
	print('Enter the metrics for evaluation (enter space separated values):\n 1. accuracy\n 2. binary_accuracy\n 3. categorical_accuracy')
	metrics=[int(x) for x in input().split(' ')]
	eval_metrics=[]
	for i in metrics:
		eval_metrics.append(metrics_dict[i])
	print('Enter the optimizer:\n 1. Adagrad\n 2. Adadelta\n 3. Adam\n 4.Nadam')
	optimizer=int(input())
	optimizer_dict={1:'Adagrad',2:'Adadelta',3:'Adam',4:'Nadam'}
	if category==1:
		classes=int(input('Select no. of output classes'))
	else:	
		classes=1
	model=return_model(input_size,hidden_units,hidden_unit_size,functions[activation_function],category,classes,eval_metrics,optimizer_dict[optimizer])
	config={'input_size':input_size,'no_of_hidden_layers':hidden_units,'hidden_unit_size':hidden_unit_size,'activation_function':functions[activation_function],'category':category,'classes':classes,'metrics':eval_metrics,'optimizer':optimizer}
	with open('config.json','w') as json_file:
		json.dump(config,json_file)	
	model.save('ann.h5')
	load_model('ann.h5')
	print('model loaded successfully')
	#print(layer.get_config())
