import numpy as np
import tensorflow as tf
import beras

from sklearn.metrics import mean_squared_error


def test_mse_forward():
    tensorflow_mse = tf.keras.losses.MeanSquaredError()
    beras_mse = beras.MeanSquaredError()

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # assert np.allclose(tensorflow_mse(x, y).numpy(), beras_mse(x, y))
    assert np.allclose(mean_squared_error(x, y), beras_mse(x, y))

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.float32)

    # assert np.allclose(tensorflow_mse(x, y).numpy(), beras_mse(x, y))
    assert np.allclose(mean_squared_error(x, y), beras_mse(x, y))
    print("MSE test passed!")

def test_leaky_relu():
    student_leaky_relu = beras.activations.LeakyReLU()
    leaky_relu = tf.keras.layers.LeakyReLU()
    test_arr = np.array(np.arange(-8,8),np.float64)
    assert(all(np.isclose(student_leaky_relu(test_arr),leaky_relu(test_arr))))
    print("Leaky ReLU test passed!")


def test_sigmoid():
    test_arr = np.array(np.arange(-8, 8),np.float64)
    student_sigmoid = beras.activations.Sigmoid()
    act_sigmoid = tf.keras.activations.sigmoid(test_arr)
    assert(all(np.isclose(student_sigmoid(test_arr),act_sigmoid)))
    print("Sigmoid test passed!")

def test_softmax():
    test_arr = np.array(np.arange(-8, 8),np.float64)
    student_softmax = beras.activations.Softmax()
    act_softmax = tf.keras.layers.Softmax()(test_arr)
    assert(all(np.isclose(student_softmax(test_arr),act_softmax)))
    print("Softmax test passed!")


def test_categorical_accuracy():
    y_true = [[0, 0, 1], [0, 1, 0]]
    y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    student_acc = beras.metrics.CategoricalAccuracy()(y_pred,y_true)
    acc = tf.keras.metrics.categorical_accuracy(y_true,y_pred)
    assert(student_acc == np.mean(acc))
    print("Categorical accuracy test passed")

def test_cce():
    y_true = np.array([[0, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    student_acc = beras.losses.CategoricalCrossentropy()(y_pred,y_true)
    #~.4556 is what your solution should output. Tensorflow's version is NOT stable so the numbers differ slightly
    assert(np.isclose(student_acc,.45561262645686385))
    print("CCE test passed")


if __name__ == "__main__":
    '''
    Uncomment the tests you would like to run for sanity checks throughout the assignment
    '''

    ### MSE Test ###
    #test_mse_forward()

    ### LeakyReLU ###
    #test_leaky_relu()

    ### Sigmoid ###
    #test_sigmoid()

    ### Softmax ###
    #test_softmax()


    ### Test Categorical Accuracy ###
    #test_categorical_accuracy()

    ### Test CCE ###
    #test_cce()
