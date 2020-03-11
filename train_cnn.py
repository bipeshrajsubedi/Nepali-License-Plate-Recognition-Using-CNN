import os
import cv2
import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical




DATA_DIR = "./dataset"
LABELS = ["0","1","2","3","4","5","6","7","8","9","10","11"]
IMG_SIZE = 50

# processing dataset images into training data
def create_train_and_test_data(DATA_DIR,LABELS):
    x = []
    y = []
    training_data = []
    for char in LABELS:
        path = os.path.join(DATA_DIR,char)
        for img in os.listdir(path):
            try:
                inp_img = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                resize_img = cv2.resize(inp_img,(IMG_SIZE,IMG_SIZE))
                training_data.append([resize_img,char])
            except Exception as e:
                pass
    random.shuffle(training_data)
    for data,label in training_data:
        x.append(data)
        y.append(label)
    x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)

    thres = 500
    x_train, x_test = x[:-thres,:],x[-thres:,:]
    y_train, y_test = y[:-thres],y[-thres:]
    print(y_train),print(y_test)
    save_data(x_train,y_train,x_test,y_test)

# saving training and test data
def save_data(x_train,y_train,x_test,y_test):
    # saving training data
    pickle_out_x = open("x_train_data.pickle", "wb")
    pickle.dump(x_train, pickle_out_x)
    pickle_out_x.close()

    pickle_out_y = open("y_train_label.pickle", "wb")
    pickle.dump(y_train, pickle_out_y)
    pickle_out_y.close()
    print(len(x_train))
    print("TRAINING DATA CREATED")

    # saving testing data
    pickle_out_x = open("x_test_data.pickle", "wb")
    pickle.dump(x_test, pickle_out_x)
    pickle_out_x.close()

    pickle_out_y = open("y_test_label.pickle", "wb")
    pickle.dump(y_test, pickle_out_y)
    pickle_out_y.close()
    print(len(x_test))
    print("TESTING DATA CREATED")

# load training and testing data
def load_data():
    x_train = pickle.load(open("x_train_data.pickle", "rb"))
    x_test = pickle.load(open("x_test_data.pickle", "rb"))
    y_train = pickle.load(open("y_train_label.pickle", "rb"))
    y_test = pickle.load(open("y_test_label.pickle", "rb"))
    return  x_train,y_train,x_test,y_test

# evaluate Training model
def check_training_model():
    x_train,y_train,x_test,y_test = load_data()
    x_train,x_test = x_train/255.0, x_test/255.0


    no_conv_layers = [2,3]
    no_nodes = [64,128]
    no_dense_layers = [0,1]

    print("TRAINING STARTED")
    for n_d_l in no_dense_layers:
        for n_n in no_nodes:
            for n_c_l in no_conv_layers:

                NAME = "{}-conv-layers-{}-nodes-{}-dense-layer".format(n_c_l,n_n,n_d_l)

                model = Sequential()
                model.add(Conv2D(n_n,(3,3),padding="same",input_shape=x_train.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

                for i in range(n_c_l-1):
                    model.add(Conv2D(n_n,(3,3),padding="same"))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Flatten())

                for j in range(n_d_l):
                    model.add(Dense(n_n))
                    model.add(Activation('relu'))

                model.add(Dense(12,activation='softmax'))

                tensorboard = TensorBoard(log_dir="./log_files_15E_32B/{}".format(NAME),
                                          histogram_freq=0,
                                          batch_size=32,
                                          write_graph=True,
                                          write_grads=False,
                                          write_images=False,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None,
                                          embeddings_data=None,
                                          update_freq='epoch')

                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                model.fit(x_train,
                          to_categorical(y_train),
                          batch_size=32,
                          epochs=15,
                          validation_data=(x_test, to_categorical(y_test)),
                          callbacks=[tensorboard]
                          )
                model.save("./MODELS/{}.model".format(NAME))
                print("{}-MODEL SAVED".format(NAME))
    print("MODEL TRAINED")

# 3-conv-128-nodes-0-dense-layer
def cnn_model():
    x_train,y_train,x_test,y_test = load_data()
    model = Sequential()

    model.add(Conv2D(128,(3,3),padding="same",input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(12,activation='softmax'))

    # metrices
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy',f1_m,precision_m,recall_m]
    )
    history = model.fit(x_train,
              to_categorical(y_train),
              batch_size=32,
              epochs=10,
              validation_data=(x_test, to_categorical(y_test)),
              )
    model.save("./MODELS/128x3.h5")
    print("{64x3}-MODEL SAVED")

    return history,model
    print("MODEL TRAINED")

# evaluation metrices

def evaluation_metrices(history,model):
    #loading test datas
    x_train,y_train,x_test,y_test = load_data()

    test_loss, test_accuracy, test_f1_score, test_precision, test_recall = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    print("loss",test_loss)
    print("acc",test_accuracy )
    print("f1_score",test_f1_score )
    print("precision",test_precision )
    print("recall", test_recall )

    # accuracy plot
    plt.title("Visualizing accuracy of 3-conv(128)-0-dense model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(history.history['accuracy'],label='train')
    plt.plot(history.history['val_accuracy'],label= 'test')
    plt.legend()
    plt.show()

    # loss plot graph
    plt.title("Visualizing loss of 3-conv(128)-0-dense model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def main():
    #create_train_and_test_data(DATA_DIR,LABELS)
    #check_training_model()
    history,model = cnn_model()
    evaluation_metrices(history,model)


if __name__ == '__main__':
    main()