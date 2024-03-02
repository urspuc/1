import numpy as np
import tensorflow as tf
import assignment

from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from preprocess import preprocess_data

def test_get_simplest_model_props():
	model_args = assignment.get_simplest_model_components()

	assert model_args.model != None
	assert model_args.batch_size != None
	assert model_args.epochs != None
	print("Simplest model props test passed!")

def test_get_simplest_model_return():
	model_args = assignment.get_simplest_model_components()

	fake_data = np.zeros((353, 784))

	assert model_args.model(fake_data).shape == (353, 10)
	print("Simplest model return shape test passed!")

def test_get_simple_model_props():
	model_args = assignment.get_simple_model_components()

	assert model_args.model != None
	assert model_args.batch_size != None
	assert model_args.epochs != None
	print("Simple model props test passed!")

def test_get_simple_model_return():
	model_args = assignment.get_simple_model_components()

	fake_data = np.zeros((353, 784))

	assert model_args.model(fake_data).shape == (353, 10)
	print("Simple model return shape test passed!")

def test_get_adv_model_props():
	model_args = assignment.get_advanced_model_components()

	assert model_args.model != None
	assert model_args.batch_size != None
	assert model_args.epochs != None
	print("Advanced model props test passed!")

def test_get_advanced_model_return():
	model_args = assignment.get_advanced_model_components()

	fake_data = np.zeros((353, 784))

	assert model_args.model(fake_data).shape == (353, 10)
	print("Advanced model return shape test passed!")


if __name__ == "__main__":
	'''
	Uncomment the tests you would like to run for sanity checks throughout the assignment
	'''

	### Simplest Model has props ###
	# test_get_simplest_model_props()

	### Simplest Model return shape ###
	# test_get_simplest_model_return()

	### Simple Model has props ###
	# test_get_simple_model_props()

	### Simple Model return shape ###
	# test_get_simple_model_return()

	### Advanced Model has props ###
	# test_get_adv_model_props()

	### Advanced Model return shape ###
	# test_get_advanced_model_return()
