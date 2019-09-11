from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import load_model
from telegram_bot_callback import TelegramBotCallback
from dl_bot import DLBot

def train_model(model,epochs):
	telegram_token = "TOKEN" 
	telegram_user_id = None   
	bot = DLBot(token=telegram_token, user_id=telegram_user_id)
	telegram_callback = TelegramBotCallback(bot)
	
	(X_train,y_train),(X_test,y_test)=mnist.load_data()
	X_train=tf.keras.utils.normalize(X_train,axis=1)
	X_test=tf.keras.utils.normalize(X_test,axis=1)
	
	model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),verbose=1,callbacks=[telegram_callback])
	
	score = model.evaluate(x_test, y_test, verbose=0)
	bot.send_message('Test loss:' + str(score[0]))
	bot.send_message('Test accuracy:' + str(score[1]))
	
