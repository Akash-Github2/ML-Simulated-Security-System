import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ImgPreprocessing as ip

numOpen = 164
numClose = 55
#New Image dimensions
wid = 40


def trainModel():
    train_images = []
    train_labels = []

    #Open
    for i in range(1, numOpen + 1):
        temp = cv2.imread(f"TrainingData/Open/img{i}.png")
        newImg = ip.resizeImg(temp, wid, wid)
        train_images.append(newImg)
        train_labels.append(1)

    # Close
    for i in range(numOpen + 1, numOpen + numClose + 1):
        temp = cv2.imread(f"TrainingData/Close/img{i}.png")
        newImg = ip.resizeImg(temp, wid, wid)
        train_images.append(newImg)
        train_labels.append(0)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(train_images.shape)
    print(len(train_labels))

    train_images = train_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(wid, wid)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=15)
    return model

def makePrediction(model, img):
    test_images = [img] #will only contain 1 at a time bc it will be doing this on the spot
    test_images = np.array(test_images)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    print("Pred List:", predictions[0])
    print("Max:", np.argmax(predictions[0]))

    return np.argmax(predictions[0]) == 1 #True if it's open, false if closed
