from numpy import mean
from numpy import std
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


def load_data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print("Train: X={0}, y={1}".format(train_x.shape, train_y.shape))
    print("Test: X={0}, y={1}".format(test_x.shape, test_y.shape))

    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    # plt.show()

    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    return train_x, train_y, test_x, test_y


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm /= 255.0
    test_norm /= 255.0

    return train_norm, test_norm


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # Compile Model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(data_x, data_y, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(data_x):
        model = define_model()
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # Fitting the model
        history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0)
        # Evaluate the model
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)

    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.title("Cross Entropy Loss")
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.legend(loc='upper left')

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.title("Classification Accuracy")
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.legend(loc='upper left')
    plt.show()


def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def run_test_harness():
    train_x, train_y, test_x, test_y = load_data()
    train_x, test_x = prep_pixels(train_x, test_x)
    model = define_model()
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=0)
    model.save('final_model.h5')
    # scores, histories = evaluate_model(train_x, train_y)
    # summarize_diagnostics(histories)
    # summarize_performance(scores)


if __name__ == '__main__':
    run_test_harness()
