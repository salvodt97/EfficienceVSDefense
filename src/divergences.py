import os
from tkinter import font
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf, warnings, tqdm, matplotlib.pyplot as plt, json5, numpy as np


from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category = NumbaPerformanceWarning)
from predict_general import general_predict
from transform_input import transform_image




def evaluate_kl_divergence(model, model_type, image):
    prediction = general_predict(model, image, model_type)
    transformed_images = transform_image(image)
    divergences = [0 for _ in range(6)]
    for index, tr_image in enumerate(transformed_images):
        prediction_tr = general_predict(model, tr_image, model_type)
        kl_divergence = max(tf.keras.losses.KLDivergence()(prediction, prediction_tr).numpy(), tf.keras.losses.KLDivergence()(prediction_tr, prediction).numpy())
        divergences[index] = float(kl_divergence)
    
    return divergences


def compute_white_box_divergences(model, quantized_model, axc_model, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images, x_test, nimages, json5_file):
    divergences = ['blurred', 'translated', 'rotated', 'flipped', 'scaled', 'contrast']
    original_h5_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_h5_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    original_quant_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    original_axc_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_quant_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_axc_divergences = {div: [0 for _ in range(nimages)] for div in divergences}

    for index, image, h5_adv_image, quantized_adv_image, axc_adv_image in tqdm.tqdm(zip(range(nimages), x_test[0:nimages], h5_adversarial_images[0:nimages], tflite_adversarial_images[0:nimages], axc_adversarial_images[0:nimages]), total = nimages, desc = "Computing White-Box Divergences..."):      
        original_h5_divergences_values = evaluate_kl_divergence(model, 0, image)   
        adv_h5_divergences_values = evaluate_kl_divergence(model, 0, h5_adv_image)       
        original_quant_divergences_values = evaluate_kl_divergence(quantized_model, 1, image)
        adv_quant_divergences_values = evaluate_kl_divergence(quantized_model, 1, quantized_adv_image)
        original_axc_divergences_values = evaluate_kl_divergence(axc_model, 2, image)    
        adv_axc_divergences_values = evaluate_kl_divergence(axc_model, 2, axc_adv_image)
        
        for i, div in enumerate(divergences):
            original_h5_divergences[div][index] = original_h5_divergences_values[i]
            original_quant_divergences[div][index] = original_quant_divergences_values[i]
            original_axc_divergences[div][index] = original_axc_divergences_values[i]
            adv_h5_divergences[div][index] = adv_h5_divergences_values[i]
            adv_quant_divergences[div][index] = adv_quant_divergences_values[i]
            adv_axc_divergences[div][index] = adv_axc_divergences_values[i]
            
    results = {'original_h5_divergences': original_h5_divergences, 'adv_h5_divergences': adv_h5_divergences, 'original_quant_divergences': original_quant_divergences, 'adv_quant_divergences': adv_quant_divergences, 'original_axc_divergences': original_axc_divergences, 'adv_axc_divergences': adv_axc_divergences}

    with open(json5_file, 'w') as f:
        json5.dump(results, f)
            
    return 



def compute_black_box_divergences(model, quantized_model, axc_model, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images, h5_x_test, tflite_x_test, axc_x_test, nimages, json5_file):
    divergences = ['blurred', 'translated', 'rotated', 'flipped', 'scaled', 'contrast']
    original_h5_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_h5_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    original_quant_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    original_axc_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_quant_divergences = {div: [0 for _ in range(nimages)] for div in divergences}
    adv_axc_divergences = {div: [0 for _ in range(nimages)] for div in divergences}

    for index, h5_image, h5_adv_image, tflite_image, quantized_adv_image, axc_image, axc_adv_image in tqdm.tqdm(zip(range(nimages), h5_x_test[0:nimages], h5_adversarial_images[0:nimages], tflite_x_test[0:nimages], tflite_adversarial_images[0:nimages], axc_x_test[0:nimages], axc_adversarial_images[0:nimages]), total = nimages, desc = "Computing Black-Box Divergences..."):
        original_h5_divergences_values = evaluate_kl_divergence(model, 0, h5_image)
        adv_h5_divergences_values = evaluate_kl_divergence(model, 0, h5_adv_image)
        original_quant_divergences_values = evaluate_kl_divergence(quantized_model, 1, tflite_image)
        adv_quant_divergences_values = evaluate_kl_divergence(quantized_model, 1, quantized_adv_image)
        original_axc_divergences_values = evaluate_kl_divergence(axc_model, 2, axc_image)
        adv_axc_divergences_values = evaluate_kl_divergence(axc_model, 2, axc_adv_image)
        
        for i, div in enumerate(divergences):
            original_h5_divergences[div][index] = original_h5_divergences_values[i]
            original_quant_divergences[div][index] = original_quant_divergences_values[i]
            original_axc_divergences[div][index] = original_axc_divergences_values[i]
            adv_h5_divergences[div][index] = adv_h5_divergences_values[i]
            adv_quant_divergences[div][index] = adv_quant_divergences_values[i]
            adv_axc_divergences[div][index] = adv_axc_divergences_values[i]
            
    results = {'original_h5_divergences': original_h5_divergences, 'adv_h5_divergences': adv_h5_divergences, 'original_quant_divergences': original_quant_divergences, 'adv_quant_divergences': adv_quant_divergences, 'original_axc_divergences': original_axc_divergences, 'adv_axc_divergences': adv_axc_divergences}

    with open(json5_file, 'w') as f:
        json5.dump(results, f)
            
    return 


def plot_divergences(json5_file, figure):
    with open(json5_file, 'r') as f:
        data = json5.load(f)

    original_h5_divergences = data['original_h5_divergences']
    adv_h5_divergences = data['adv_h5_divergences']
    original_quant_divergences = data['original_quant_divergences']
    adv_quant_divergences = data['adv_quant_divergences']
    original_axc_divergences = data['original_axc_divergences']
    adv_axc_divergences = data['adv_axc_divergences']
    divergences = ['blurred', 'translated', 'rotated', 'flipped', 'scaled', 'contrast']
    plot_div = ['Blurring', 'Translation', 'Rotation', 'Flipping', 'Scaling', 'Contrast']
    fig, axs = plt.subplots(3, 2, figsize=(22, 30))

    labels = ["CNN\nclean data", "CNN\nadv data", "QNN\nclean data", "QNN\nadv data", "AxNN\nclean data", "AxNN\nadv data"]

    for (i, div), plt_div  in zip(enumerate(divergences), plot_div):
        row = i // 2
        col = i % 2
        # data_to_plot = [original_h5_divergences[div], adv_h5_divergences[div], original_quant_divergences[div], adv_quant_divergences[div], original_axc_divergences[div], adv_axc_divergences[div]]
        data_to_plot = [
            [x for x in original_h5_divergences[div] if x != 0],
            [x for x in adv_h5_divergences[div] if x != 0],
            [x for x in original_quant_divergences[div] if x != 0],
            [x for x in adv_quant_divergences[div] if x != 0],
            [x for x in original_axc_divergences[div] if x != 0],
            [x for x in adv_axc_divergences[div] if x != 0]
        ]
        # data_to_plot = [np.array(original_h5_divergences[div]) + epsilon, np.array(adv_h5_divergences[div]) + epsilon, np.array(original_quant_divergences[div]) + epsilon, np.array(adv_quant_divergences[div]) + epsilon, np.array(original_axc_divergences[div]) + epsilon, np.array(adv_axc_divergences[div]) + epsilon]
        axs[row, col].boxplot(data_to_plot)
        axs[row, col].set_title(plt_div, fontsize = 40, weight = 'bold')
        axs[row, col].set_xticklabels(labels, fontsize = 22)
        
        axs[row, col].set_yticklabels(axs[row, col].get_yticks(), fontsize = 25)
        axs[row, col].set_yscale('log')

    if os.path.exists(figure):
        os.remove(figure)
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()


def plot_different_divergences(json5_file, output_dir):
    with open(json5_file, 'r') as f:
        data = json5.load(f)

    original_h5_divergences = data['original_h5_divergences']
    adv_h5_divergences = data['adv_h5_divergences']
    original_quant_divergences = data['original_quant_divergences']
    adv_quant_divergences = data['adv_quant_divergences']
    original_axc_divergences = data['original_axc_divergences']
    adv_axc_divergences = data['adv_axc_divergences']
    divergences = ['blurred', 'translated', 'rotated', 'flipped', 'scaled', 'contrast']
    plot_div = ['Blurring', 'Translation', 'Rotation', 'Flipping', 'Scaling', 'Contrast']

    labels = ["FpNN\nclean data", "FpNN\nadv data", "QNN\nclean data", "QNN\nadv data", "AxNN\nclean data", "AxNN\nadv data"]

    for div, plt_div in zip(divergences, plot_div):
        data_to_plot = [
            [x for x in original_h5_divergences[div] if x != 0],
            [x for x in adv_h5_divergences[div] if x != 0],
            [x for x in original_quant_divergences[div] if x != 0],
            [x for x in adv_quant_divergences[div] if x != 0],
            [x for x in original_axc_divergences[div] if x != 0],
            [x for x in adv_axc_divergences[div] if x != 0]
        ]

        plt.figure(figsize=(12, 9))
        plt.boxplot(data_to_plot)
        # plt.title(plt_div, fontsize=40, weight='bold')
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, fontsize=22)
        plt.yticks(fontsize=25)
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()

        figure = os.path.join(output_dir, f'{div}.pdf')
        if os.path.exists(figure):
            os.remove(figure)
        plt.savefig(figure)
        plt.close()
