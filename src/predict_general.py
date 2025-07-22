from audioop import add
from re import sub
import numpy as np, tflite
from scipy.special import softmax

def general_predict(model, image, model_type):
    if model_type == 0:
        return model.predict(np.expand_dims(image, axis=0), verbose = 0)
    elif model_type == 1:
        return predict_quantized_model(model, image)
    elif model_type == 2:
        return predict_axc(model, image)
    else:
        raise ValueError("Invalid model_type. Please provide a valid model_type value (0, 1, or 2).")

    
def predict_axc(axc_model_struct, image):
    axc_model = axc_model_struct["axc_model"]
    axc_model.net.predict(image)
    output = axc_model.net.layers[-2].results.copy_to_host()
    add_parameter = axc_model_struct["add_par"]
    mul_aparemeter = axc_model_struct["mul_par"]
    output = (output + add_parameter) * mul_aparemeter

    return softmax(output)


def predict_quantized_model(model_quantized, image):
    interpreter = model_quantized[0]
    root = model_quantized[1]
    subgraph = root.Subgraphs(0)
    indices = np.arange(0, root.Subgraphs(0).OperatorsLength())
    add_parameter = interpreter.get_tensor_details()[subgraph.Operators(indices[-1]).Inputs(0)]["quantization"][1]
    mul_aparemeter = interpreter.get_tensor_details()[subgraph.Operators(indices[-1]).Inputs(0)]["quantization"][0]
    image = np.expand_dims(image, axis=0).astype(np.float32)
    # print(image.shape)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
    interpreter.invoke()
    output = interpreter.get_tensor(subgraph.Operators(indices[-3]).Outputs(0))
    output = (output + add_parameter) * mul_aparemeter
    
    
    return softmax(output)