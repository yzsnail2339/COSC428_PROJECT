## 🛣️ Drive Road Surface Detection using U²-Net

This project implements a deep learning-based road surface recognition and segmentation method using **U²-Net** with **Residual U-blocks (RSU)**. The system efficiently segments road surfaces from high-resolution images while distinguishing related background objects like vehicles, traffic signs, and vegetation — aiming to support autonomous driving and road infrastructure applications.

### 📌 Highlights

* 🚗 Real-time binary segmentation of drivable lanes
* 🧠 Utilizes **U²-Net** architecture with **RSU** for multi-scale feature learning
* 🪶 Lightweight variant (U²Net Lite) for resource-constrained environments
* 🎯 Achieves high segmentation performance:
  **MAE**: 3.9%, **Max F1**: 87.5%, **IoU**: 80.1%

---

### 🧠 Model Architecture

The core model is [**U²-Net**](https://doi.org/10.1016/j.patcog.2020.107404), a nested U-shaped architecture that enhances detail preservation and global context through:

* **RSU blocks** for multi-scale contextual encoding
* Deep hierarchical encoder-decoder layers without pre-trained backbone
* Lightweight version with reduced channel widths

> Full and Lite configurations are supported.

---

### 📁 Dataset

* **Source**: [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
* **Preprocessing**:

  * Converted 124-class segmentation to **binary** (road vs. background)
  * Resized & randomly cropped to `320x320`
  * Data augmentation: flipping, perspective distortion, normalization

---

### ⚙️ Training Setup

* **Framework**: PyTorch 2.2.1 + CUDA 12.1
* **GPU**: NVIDIA RTX 3070 Laptop
* **Epochs**: 350
* **Batch Size**: 8
* **Optimizer**: Adam, learning rate decay
* **Loss Function**: Multi-output weighted binary cross-entropy

---

### 📊 Evaluation Metrics

| Metric | Description                                   |
| ------ | --------------------------------------------- |
| MAE    | Mean Absolute Error per pixel                 |
| MaxF1  | Max F1-score over thresholds                  |
| IoU    | Intersection over Union (road vs. background) |

---

### 🏁 Results

| Model       | MAE   | MaxF1 | IoU   | Size    |
| ----------- | ----- | ----- | ----- | ------- |
| U²-Net Full | 0.039 | 0.875 | 0.801 | 504 MB  |
| U²-Net Lite | 0.045 | 0.868 | 0.788 | 13.4 MB |

> Both models showed strong generalization on high-resolution road images, but the **Lite** model offers significant size and deployment benefits.

---

### 🖼️ Sample Predictions

Original → Prediction
![sample\_road\_segmentation](path/to/your/image.png)

---

### 🚧 Limitations

* Model struggles in **nighttime**, **aerial**, or **low-contrast** scenes
* Poor performance on **multi-class** road segmentation tasks

---

### 🔭 Future Work

* Incorporate **weather conditions** (rain, snow) into training
* Add **attention mechanisms** (e.g., ECA-AS) for refinement
* Try **temporal modeling** for video-based segmentation (e.g., TCNs)
* Explore **multi-modal** input (e.g., depth maps or LiDAR)

---

### 📎 Citation

If you use this work, please cite:

```
@misc{yang2025road,
  title={Drive Road Surface Detection using U²-Net},
  author={Zhun Yang},
  year={2025},
  note={University of Canterbury, COSC428 Project}
}
