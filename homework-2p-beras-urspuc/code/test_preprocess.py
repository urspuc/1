from preprocess import load_and_preprocess_data
from beras.core import Tensor

def test_preprocess_type():
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()

    assert type(X_train) is Tensor
    assert type(X_test) is Tensor
    assert type(Y_train) is Tensor
    assert type(Y_test) is Tensor
    print("Preprocess types test passed!")

def test_preprocess_shapes():
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    assert X_train.shape == (60000, 784)
    assert X_test.shape == (10000, 784)
    assert Y_train.shape == (60000,)
    assert Y_test.shape == (10000,)
    print("Preprocess shapes test passed!")


if __name__ == '__main__':
    '''
    Uncomment the tests you would like to run for sanity checks throughout the assignment
    '''

    ### preprocess type ###
    # test_preprocess_type()

    ### preprocess shapes ###
    # test_preprocess_shapes()
    