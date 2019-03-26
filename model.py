import keras
from pretrained.wideResNet import WideResNet
from keras.layers.core import Dense
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_4 = Input(shape=(3, 50, 50), name='Input_4')
	WideResNet_2_model = WideResNet(name='WideResNet_2', input_tensor = Input_4)
	WideResNet_2 = WideResNet_2_model(Input_4)
	aliases['WideResNet_2'] = WideResNet_2_model.name
	GlobalMaxPooling2D_9 = GlobalMaxPooling2D(name='GlobalMaxPooling2D_9')(WideResNet_2)
	Dense_5 = Dense(name='Dense_5',output_dim= 20,activation= 'relu' )(GlobalMaxPooling2D_9)
	Dense_6 = Dense(name='Dense_6',output_dim= 1,activation= 'sigmoid' )(Dense_5)

	model = Model([Input_4],[Dense_6])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adagrad()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'binary_crossentropy'

def get_batch_size():
	return 512

def get_num_epoch():
	return 20

def get_data_config():
	return '{"shuffle": false, "samples": {"split": 1, "test": 0, "validation": 55504, "training": 222019}, "datasetLoadOption": "full", "numPorts": 1, "mapping": {"label": {"type": "Numeric", "port": "OutputPort0", "shape": "", "options": {"Scaling": 1, "Normalization": false}}, "filename": {"type": "Image", "port": "InputPort0", "shape": "", "options": {"vertical_flip": false, "Augmentation": false, "pretrained": "None", "height_shift_range": 0, "Resize": true, "Scaling": 1, "horizontal_flip": false, "Height": "50", "shear_range": 0, "Width": "50", "width_shift_range": 0, "rotation_range": 0, "Normalization": false}}}, "kfold": 1, "dataset": {"name": "lung_cancer", "type": "private", "samples": 277524}}'