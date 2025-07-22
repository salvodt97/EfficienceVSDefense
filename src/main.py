import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse, numpy as np, tensorflow as tf, warnings, tflite, tqdm
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter('ignore', category = NumbaPerformanceWarning)

from generic_functions import load_dataset, limit_resource_usage, retrun_indices, return_black_box_indices
from manage_tflite import approximate_model
from divergences import compute_white_box_divergences, plot_divergences, compute_black_box_divergences, plot_different_divergences



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",               "-m", type = str, help = "Model to attack. Please specify also it's relative path.",           required = True)
    parser.add_argument("--quantized_model",     "-q", type = str, help = "Quantized Model to attack. Please specify also it's relative path.", required = True)
    parser.add_argument("--analysis_path",       "-p", type = str, help = "Path which contains data to analyze",                                required = True)
    parser.add_argument("--nimages",             "-n", type = int, help = "Number of images to analyze",                                        required = True)
    parser.add_argument("--plot",                "-l", help = "If passed, only the plot is generated",                                          required = False, action = "store_true")
    args = parser.parse_args()

    limit_resource_usage()
    
    # model = tf.keras.models.load_model(args.model)
    # model_name = args.model.split("/")[-1].split(".")[0]
    # model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics=['accuracy'])
    # model_quantized = tf.lite.Interpreter(model_path = args.quantized_model, experimental_preserve_all_tensors = True)
    # model_quantized.allocate_tensors()
    # f = open(args.quantized_model,"rb")
    # buffer = f.read()
    # root = tflite.Model.GetRootAsModel(buffer, 0)
    # quantized_model = [model_quantized, root]
    # axc_model = approximate_model(args.quantized_model, model_name, 1)
    
    h5_adversarial_images       = np.load(os.path.join(args.analysis_path, "h5_images.npy"))
    tflite_adversarial_images   = np.load(os.path.join(args.analysis_path, "tflite_images.npy"))
    axc_adversarial_images      = np.load(os.path.join(args.analysis_path, "axc_images.npy"))
    json5_file = os.path.join(args.analysis_path, "divergences.json5")
    
    # if "OP" in args.analysis_path:
    #     attack = "op"
    # elif "BIM" in args.analysis_path:
    #     attack = "bim"
    # elif "DeepFool" in args.analysis_path:
    #     attack = "dp"
    # elif "PGD" in args.analysis_path:
    #     attack = "pgd"
        
    # figure = os.path.join(args.analysis_path, f"divergences_{model_name.lower()}_{attack}.pdf")
    
    # _, _, _, _, x_test, _ = load_dataset(model)
    if args.plot:
            # plot_divergences(json5_file, figure)
            plot_different_divergences(json5_file, args.analysis_path)
    # else:
    #     if "OP" in args.analysis_path:
    #         h5_indices, tflite_indices, axc_indices = return_black_box_indices(os.path.join(args.analysis_path, "results.csv"))
    #         h5_x_test = x_test[h5_indices]
    #         tflite_x_test = x_test[tflite_indices]
    #         axc_x_test = x_test[axc_indices]
    #         h5_adversarial_images = h5_adversarial_images[h5_indices]
    #         tflite_adversarial_images = tflite_adversarial_images[tflite_indices]
    #         axc_adversarial_images = axc_adversarial_images[axc_indices]
    #         if len(h5_x_test) < args.nimages or len(tflite_x_test) < args.nimages or len(axc_x_test) < args.nimages:
    #             nimages = min(len(h5_x_test), len(tflite_x_test), len(axc_x_test))
    #         else:
    #             nimages = args.nimages
    #         compute_black_box_divergences(model, quantized_model, axc_model, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images, h5_x_test, tflite_x_test, axc_x_test, nimages, json5_file) 
    #     else:  
    #         images_indices = retrun_indices(os.path.join(args.analysis_path, "results.csv"))
    #         x_test = x_test[images_indices]
    #         h5_adversarial_images = h5_adversarial_images[images_indices]
    #         tflite_adversarial_images = tflite_adversarial_images[images_indices]
    #         axc_adversarial_images = axc_adversarial_images[images_indices]
    #         if len(h5_adversarial_images) < args.nimages:
    #             nimages = len(x_test[images_indices])
    #         else:    
    #             nimages = args.nimages
    #         compute_white_box_divergences(model, quantized_model, axc_model, h5_adversarial_images, tflite_adversarial_images, axc_adversarial_images, x_test, nimages, json5_file)
    #     plot_divergences(json5_file, figure)



    
   