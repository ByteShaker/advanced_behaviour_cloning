import os
import sys
import pickle
import pandas

from PIL import Image
import numpy as np

import cv2

from sklearn.model_selection import train_test_split

def cut_images_to_arr(img_Center):
    arr_Center = np.array(img_Center)
    arr_Center = arr_Center[50:]
    return arr_Center

def convert_to_HLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls


def normalize_image(img):
    max = 255. #np.max(img)
    return (((img) / max) - 0.5)

if __name__ == '__main__':

    dataframe = pandas.read_csv('./driving_log.csv', header=None)
    driving_log = dataframe.values

    X_train = []
    y_train = []

    for el in driving_log:
        img_Center = Image.open(el[0])
        cut_img = cut_images_to_arr(img_Center)
        #arr = np.array(cut_img)
        #arr = arr[50:]

        hls = convert_to_HLS(cut_img)

        norm_hls = normalize_image(hls)

        #img = Image.fromarray(np.uint8(hls[:,:,2]))
        #img.show()
        X_train.append(norm_hls)
        y_train.append(el[3])


    X_train = np.array(X_train)
    #X_train = normalize_image(X_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2)

    print('Start Pickling Data')

    pickle_data = pickle.dumps(
        {
            'train_dataset': X_train,
            'train_labels': y_train,
            'val_dataset': X_val,
            'val_labels': y_val
        }, pickle.HIGHEST_PROTOCOL)

    del X_train, X_val, y_train, y_val

    print('Data Pickled')

    pickle_size = sys.getsizeof(pickle_data)
    print(pickle_size)

    # Save the data for easy access
    pickle_file = 'train_data_level3.pickle'
    exists = False

    max_bytes = 2 ** 31 - 1

    while not exists:
        if not os.path.isfile(pickle_file):
            print('Pickle Train_data')
            try:
                with open(pickle_file, 'wb') as p_train_data:
                    for idx in range(0, pickle_size, max_bytes):
                        p_train_data.write(pickle_data[idx:idx + max_bytes])

            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise

            print('Train_data in Pickle File.')
            exists = True
        else:
            print("Pickle Filename already in use. Choose another name: *.pickle")
            pickle_file = input("Enter: ")