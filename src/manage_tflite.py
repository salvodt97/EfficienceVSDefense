import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflite, tensorflow as tf, numpy as np, json5, inspectnn, pandas as pd, tqdm
from inspectnn.Model.GenericModel_tflite import GenericModelTflite
from scipy.special import softmax


def return_axc__final_layer_output(axc_model_struct, image):
    axc_model = axc_model_struct["axc_model"]
    axc_model_struct["axc_model"].net.predict(image)[0]
    output = axc_model.net.layers[-2].results.copy_to_host()
        
    return output


def return_quantized_network(quantized_model_path):
    interpeter = tf.lite.Interpreter(model_path = quantized_model_path, experimental_preserve_all_tensors = True)
    interpeter.allocate_tensors()
    f = open(quantized_model_path,"rb")
    buffer = f.read()
    root = tflite.Model.GetRootAsModel(buffer, 0)
    
    return interpeter, root

def take_dequantized_quantization_parameter(interpeter, buffer):
    root = tflite.Model.GetRootAsModel(buffer, 0)
    subgraph = root.Subgraphs(0)
    indices = np.arange(0, root.Subgraphs(0).OperatorsLength())
    add_parameter = interpeter.get_tensor_details()[subgraph.Operators(indices[-1]).Inputs(0)]["quantization"][1]
    mul_aparemeter = interpeter.get_tensor_details()[subgraph.Operators(indices[-1]).Inputs(0)]["quantization"][0]
    
    return add_parameter, mul_aparemeter

# def take_quantization_parameters(interpeter, buffer):
#     root = tflite.Model.GetRootAsModel(buffer, 0)
#     subgraph = root.Subgraphs(0)
#     op = subgraph.Operators(subgraph.OperatorsLength()-2) 
#     add_parameter = interpeter.get_tensor_details()[op.Inputs(0)]["quantization"][1]
#     mul_aparemeter = interpeter.get_tensor_details()[op.Inputs(0)]["quantization"][0]
    
#     return add_parameter, mul_aparemeter


def quantize_network(quantized_model_path):
    interpeter = tf.lite.Interpreter(model_path = quantized_model_path, experimental_preserve_all_tensors = True)
    interpeter.allocate_tensors()
    f = open(quantized_model_path,"rb")
    buf = f.read()
    add_parameter, mul_aparemeter = take_dequantized_quantization_parameter(interpeter, buf)
    
    return interpeter, buf, add_parameter, mul_aparemeter


def take_muls_details(model_name, muls_details, file = "../../parameters.json"):
    with open(file, 'r') as f:
        json_file = json5.load(f)
    list_mux = json_file[model_name][f'list_multipliers{str(muls_details)}']
    
    return list_mux


def approximate_model(quantized_model, model_name, muls_details, path_multipliers = "../../../pyALS-lenet5-int8/AxMult/EvoApproxLite/ALWANN", muls_conf_file = "../../mulconfigurations.json5", file = "../../parameters.json"):
    configuration = json5.load(open(muls_conf_file))
    
    _, _, add_par, mul_par = quantize_network(quantized_model)
    list_mux = take_muls_details(model_name, muls_details, file)
    axc_model = GenericModelTflite(quantized_model, False)
    axc_model.load_all_multiply(path_multipliers)
    if len(list_mux) > 1:
        axc_model.net.update_multipler(axc_model.generate_multipler_list(list_mux))
    else:
        axc_model.net.update_multipler([axc_model.all_multiplier[list_mux]])
        
    mults_per_layer = np.array([ i.n_moltiplicazioni for i in axc_model.net.layers if isinstance(i, (inspectnn.Conv.ConvLayer.ConvLayer, inspectnn.Dense.DenseLayer.DenseLayer))])
    power_per_layer = [mul["power"] for net_mul in axc_model.generate_multipler_list(list_mux) for mul in configuration["multipliers"] if mul["path"].split("/")[-1].split(".")[0] in str(net_mul).split(".")[-1].split(" ")[0]]
    area_per_layer = [mul["area"] for net_mul in axc_model.generate_multipler_list(list_mux) for mul in configuration["multipliers"] if mul["path"].split("/")[-1].split(".")[0] in str(net_mul).split(".")[-1].split(" ")[0]]  
    baseline_power = np.dot([configuration["multipliers"][0]["power"]] * len(axc_model.generate_multipler_list(list_mux)), mults_per_layer) / 65536   
    baseline_area = np.sum([configuration["multipliers"][0]["area"]] * len(axc_model.generate_multipler_list(list_mux)))
    approx_power = np.dot(power_per_layer, mults_per_layer) / 65536
    approx_area = np.sum(area_per_layer)
    floating_point_power = 2.048 * np.sum(mults_per_layer) / 65536
    floating_point_area = 1215.088 * len(axc_model.generate_multipler_list(list_mux))
    
    axc_model_struct = {}
    axc_model_struct["axc_model"]    = axc_model
    axc_model_struct["add_par"]      = add_par
    axc_model_struct["mul_par"]      = mul_par
    axc_model_struct["baseline_power"]   = baseline_power
    axc_model_struct["baseline_area"]    = baseline_area
    axc_model_struct["axc_power"]   = approx_power
    axc_model_struct["axc_area"]    = approx_area
    axc_model_struct["floating_point_area"]    = floating_point_area
    axc_model_struct["floating_point_power"]    = floating_point_power
    
    return axc_model_struct


