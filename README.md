# Time Series Anomaly Detection using Association Discrepancy

This project is a PyTorch-based reimplementation of the **Anomaly Transformer** proposed in the paper *"Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"* (Xu et al., ICLR 2022). It detects anomalies in time-series data by modeling association discrepancies using attention mechanisms. Our version is simplified to work with smaller datasets on limited hardware, and includes Streamlit integration for real-time inference and visualization.

---

## About the Original Work

The Anomaly Transformer introduces a novel concept called **association discrepancy**, which measures the difference between attention-derived association patterns in normal vs. abnormal data. Anomalies tend to have highly localized attention, and this behavior is captured and used as a detection signal. The model sets new benchmarks on several anomaly detection datasets.

Original Paper: https://arxiv.org/abs/2110.02642  
Official GitHub: https://github.com/thuml/Anomaly-Transformer

---

## What We Did

- Studied the original Anomaly Transformer paper and architecture in-depth.
- Reimplemented the model using **PyTorch**, making it compatible with minimal datasets that can run on commodity hardware (e.g., 4GB VRAM GPU).
- Implemented and tested an **RNN-based LSTM autoencoder** for comparison.
- Gained strong foundational knowledge of **Transformers**, **Self-Attention**, and how they adapt to time series.
- Integrated a **Streamlit dashboard** (external module) for anomaly score visualization and real-time interaction.
- Trained the model using a subset of the **Server Machine Dataset (SMD)** and tested on a small sample for performance evaluation.

---

## Repository Structure

```
.
├── AnomalyAttention.py               # Implements the anomaly-aware self-attention with discrepancy computation
├── AnomalyTransformer.py            # Defines the Anomaly Transformer model architecture
├── anomaly_transformer_weights.pth  # Pretrained model weights
├── datasets/
│   └── ServerMachineDataset/        # Contains sample SMD time-series data (CSV format)
├── evaluate.py                      # Evaluation and inference script
├── train.py                         # Model training script
└── __pycache__/                     # Python bytecode cache (ignore)
```

---

## How to Run

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch numpy pandas matplotlib scikit-learn
```

### 2. Train the Model

```bash
python train.py
```

Make sure your dataset is placed in `datasets/ServerMachineDataset/`. You can modify the script to change window size, epochs, or dataset path.

### 3. Evaluate the Model

```bash
python evaluate.py
```

This script will compute the anomaly scores, compare with ground truth (if provided), and visualize the results.

---

## Dataset Used

We use a minimal version of the **Server Machine Dataset (SMD)** due to hardware constraints. The dataset contains multivariate sensor readings from industrial machines. The files are located in:

```
datasets/ServerMachineDataset/
```

You can substitute your own CSV-format time-series files with timestamps and values in similar shape.

---

## Learning Outcomes

Through this project, we gained hands-on knowledge of:

- How **self-attention** works and how it's adapted for time series data.
- The architecture of **Transformer encoders** for unsupervised tasks.
- The importance of **association discrepancy** in anomaly localization.
- Implementing and training models in **PyTorch** from scratch.
- Comparing Transformer-based detection with **RNN (LSTM)** baselines.
- Building an interactive UI with **Streamlit** for real-time anomaly detection.

---

## Future Extensions

- Train on full-scale SMD, SMAP, and MSL datasets with longer sequences.
- Integrate online (streaming) anomaly detection with real-time windowing.
- Add **Streamlit integration** directly into this repo for visualization.
- Experiment with other models: OmniAnomaly, USAD, DTAAD, Informer, etc.
- Package the model as a REST API for deployment in monitoring systems.

---

## Citation

If you use this work or build upon it, please cite the original authors:

```bibtex
@inproceedings{xu2022anomaly,
  title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author={Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

---

## License

This project is for educational use. Refer to the [Anomaly Transformer GitHub repository](https://github.com/thuml/Anomaly-Transformer) for original license terms.
