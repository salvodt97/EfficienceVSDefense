import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf, numpy as np, pandas as pd, math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from predict_general import general_predict




def standardize(vector):
    vector = np.array(vector)
    min_val = np.min(vector)
    max_val = np.max(vector)
    if not np.isclose(min_val, max_val, rtol=1e-15, atol=1e-15):
        vector = (vector - min_val) / (max_val - min_val)
    else:
        vector = np.clip(vector, 0, 1)
    return vector


def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.maximum(a, b)))
    mse = np.mean((np.nan_to_num(np.array(a, dtype=np.float32)) - np.nan_to_num(np.array(b, dtype=np.float32))) ** 2)
    if mse == 0:
        return 0, 100
    psnr = 20 * np.log10(max_ab / (np.sqrt(mse)))
    return mse, psnr


def get_mare(a, b):
    absolute_error = np.abs(a - b)
    mare_results = np.zeros_like(a, dtype=float)
    nonzero_indices = np.where(a != 0)
    mare_results[nonzero_indices] = absolute_error[nonzero_indices] / a[nonzero_indices]
    
    return np.mean(mare_results)


def retrun_indices(file):
    dataframe = pd.read_csv(file)
    indices = []
    [indices.append(row['Image']) for _, row in dataframe.iterrows()]
    return indices

def return_black_box_indices(file):
    dataframe = pd.read_csv(file)
    h5_indices = []
    tflite_indices = []
    axc_indices = []
    [h5_indices.append(row['Image']) for _, row in dataframe.iterrows() if row['h5_fooled'] == 1]
    [tflite_indices.append(row['Image']) for _, row in dataframe.iterrows() if row['tflite_fooled'] == 1]
    [axc_indices.append(row['Image']) for _, row in dataframe.iterrows() if row['axc_fooled'] == 1]
    return h5_indices, tflite_indices, axc_indices


def resnet50_preprocess_images(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims


def load_dataset(model):
    if model.input_shape == (None, 32, 32, 3):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
    elif model.input_shape == (None, 28, 28, 1):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    if any("resnet50" in layer.name for layer in model.layers):
        x_train = resnet50_preprocess_images(x_train)
        x_test = resnet50_preprocess_images(x_test)
        x_val = resnet50_preprocess_images(x_val)
    else: 
        x_train /= 255
        x_test /= 255
        x_val /= 255

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    

    return x_train, y_train, x_val, y_val, x_test, y_test


def evaluate_nets(model, quant_model, axc_model, dataset, labels):
    h5_accuracy = 0
    tflite_accuracy = 0
    axc_accuracy = 0
    for image, label in tqdm(zip(dataset, labels), total = len(dataset), desc="Eavluating corrected prediction..."):
        prediction = general_predict(model, image, 0)
        if np.argmax(prediction) == np.argmax(label):
            h5_accuracy += 1
        quantized_prediction = general_predict(quant_model, image, 1)
        if np.argmax(quantized_prediction) == np.argmax(label):
            tflite_accuracy += 1
        approximated_prediction = general_predict(axc_model, image, 2)
        if (np.argmax(approximated_prediction) == np.argmax(label)):
            axc_accuracy += 1
    return h5_accuracy/len(dataset) * 100, tflite_accuracy/len(dataset) * 100, axc_accuracy/len(dataset) * 100


def evaluate_nets_vs_adversarial(model, quant_model, axc_model, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images, labels):
    h5_accuracy = 0
    tflite_accuracy = 0
    axc_accuracy = 0
    for label, h5_adv_image, tflite_adv_image, axc_adv_image in tqdm(zip(labels, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images), total = len(h5_adversarial_images), desc="Eavluating Network Adversarial Accuracy..."):
        prediction = general_predict(model, h5_adv_image, 0)
        if np.argmax(prediction) == np.argmax(label):
            h5_accuracy += 1
        quantized_prediction = general_predict(quant_model, tflite_adv_image, 1)
        if np.argmax(quantized_prediction) == np.argmax(label):
            tflite_accuracy += 1
        approximated_prediction = general_predict(axc_model, axc_adv_image, 2)
        if (np.argmax(approximated_prediction) == np.argmax(label)):
            axc_accuracy += 1
    return h5_accuracy, tflite_accuracy, axc_accuracy


def evaluate_axc_model(axc_model, dataset, labels):
    axc_accuracy = 0
    for image, label in tqdm(zip(dataset, labels), total = len(dataset), desc="Eavluating AxC accuracy...", leave = False):
        approximated_prediction = general_predict(axc_model, image, 2)
        if (np.argmax(approximated_prediction) == np.argmax(label)):
            axc_accuracy += 1
    return axc_accuracy/len(dataset) * 100      
   

def return_predicted(model, quant_model, axc_model, dataset, labels, original_indices):
    correct_predictions = []
    for index, image, label in tqdm(zip(original_indices, dataset, labels), total = len(dataset), desc="Eavluating corrected prediction..."):
        image_prediction = {}
        prediction = general_predict(model, image, 0)
        quantized_prediction = general_predict(quant_model, image, 1)
        approximated_prediction = general_predict(axc_model, image, 2)
        if np.argmax(prediction) == np.argmax(label) and np.argmax(quantized_prediction) == np.argmax(label) and np.argmax(approximated_prediction) == np.argmax(label):     
            image_prediction['image'] = image
            image_prediction['index'] = index
            image_prediction['label'] = label
            image_prediction['h5_prediction'] = prediction
            image_prediction['tflite_prediction'] = quantized_prediction
            image_prediction['axc_prediction'] = approximated_prediction
            correct_predictions.append(image_prediction)
    return correct_predictions

def limit_resource_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return





