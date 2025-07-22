
# InputTransf: Input Transformations and Adversarial Attacks on Efficient CNNs

This repository contains the code and experimental results presented in the paper:

**"Ineffectiveness of Digital Transformations for Detecting Adversarial Attacks Against Quantized and Approximate CNNs"**  
*Salvatore Barone, Valentina Casola, Salvatore Della Torca*  
Presented at: **2024 IEEE International Conference on Cyber Security and Resilience (CSR)**  
📄 [DOI: 10.1109/CSR61664.2024.10679345](https://doi.org/10.1109/CSR61664.2024.10679345)

---

## 🔍 Overview

This project investigates the effectiveness of **input image transformations** (e.g., rotation, blur, contrast) to detect **adversarial attacks** on Convolutional Neural Networks (CNNs), including their **quantized (QNN)** and **approximate (AxNN)** versions.

While such transformations have shown success in detecting adversarial examples in standard (floating-point) CNNs, this work demonstrates their **ineffectiveness on efficient models** — i.e., quantized and approximate CNNs — due to the masking effect introduced by hardware-related modifications.

---

## 📁 Repository Structure

```
InputTransf/
├── src/                    # Core source code
│   ├── main.py             # Main script to run experiments
│   ├── divergences.py      # KL divergence and statistical analysis
│   ├── transform_input.py  # Image transformation logic
│   ├── manage_tflite.py    # TFLite model handling
│   └── ...
├── Results/                # Collected experimental data
│   ├── MinNet_ResultsBIM/
│   ├── MinNet_ResultsPGD/
│   └── ...                 # One folder per attack-method/model combination
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧪 Attacks and Defenses Investigated

### Networks:
- ResNet8
- ResNet24
- MinNet (custom lightweight model)

### Adversarial Attacks:
- **BIM** (Basic Iterative Method)
- **PGD** (Projected Gradient Descent)
- **DeepFool**
- **One-Pixel Attack** (Differential Evolution-based)

### Digital Transformations:
- **Topological:** rotation, flipping, translation, scaling  
- **Appearance-based:** blur, contrast

---

## 📊 Key Findings

- Input transformations (as detection tools) are effective only for **floating-point CNNs**.
- **Quantized** and **approximate** CNNs show **statistically indistinguishable KL-divergences** between clean and adversarial samples.
- This undermines the possibility of using a fixed threshold to detect adversarial inputs.
- Highlights the need for new, specialized defense methods for efficient CNNs.

---

## ⚙️ Approximate Neural Networks with InspectNN

This project uses the [**InspectNN**](https://pypi.org/project/inspectnn/) library to implement **approximate CNNs**. InspectNN enables model-level analysis and design space exploration by injecting approximate multipliers into convolutional and dense layers. Multi-objective optimization is applied to minimize the tradeoff between **accuracy loss** and **hardware resource usage**.

---

## 📦 Setup

### 1. Create Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: Some dependencies such as `cupy` and `tensorflow` may require system-specific setup (e.g., CUDA, GPU support).

---

## ▶️ Run Experiments

You can execute the full pipeline by running:

```bash
python src/main.py
```

The script orchestrates:
- Generation of adversarial examples
- Application of input transformations
- KL-divergence computation
- Statistical result logging

---

## 📜 License

This repository is licensed under the **MIT License**.  
See the [LICENSE](https://opensource.org/license/mit/) file for details.

---

## 📚 Citation

If you use this code or data, please cite our paper:

```bibtex
@INPROCEEDINGS{10679345,
  author={Barone, Salvatore and Casola, Valentina and Della Torca, Salvatore},
  booktitle={2024 IEEE International Conference on Cyber Security and Resilience (CSR)}, 
  title={Ineffectiveness of Digital Transformations for Detecting Adversarial Attacks Against Quantized and Approximate CNNs}, 
  year={2024},
  pages={290-295},
  doi={10.1109/CSR61664.2024.10679345}
}
```

## 🙏 Acknowledgments
This work was carried out at the University of Napoli Federico II, with experiments performed in collaboration with the authors' research groups.
