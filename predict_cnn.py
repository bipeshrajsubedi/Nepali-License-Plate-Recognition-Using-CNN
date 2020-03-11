import localization_segmentation
import tensorflow as tf
import cv2
import numpy as np



IMG_SIZE = 50


model = tf.keras.models.load_model('128x3.model')


roi_list = np.array(localization_segmentation.roi_list)
print(len(roi_list))

def predict_char(roi_list):
    predicted_char = []
    for img in roi_list:
        gray_img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(IMG_SIZE,IMG_SIZE))
        reshape_img = np.array(gray_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
        predict = model.predict([reshape_img])
        print(predict)
        predicted_char.append(predict)

    return predicted_char


def main():
    result = predict_char(roi_list)
    print("PLATE NO::",result)

if __name__ == '__main__':
    main()