# 🌙 Low-Light Image Enhancement using U-Net

A deep learning project that enhances low-light images using a **U-Net** architecture with a **pretrained ResNet34 encoder** from **Segmentation Models PyTorch (SMP)**. The model improves image brightness while reducing noise and preserving visual details.

---

## 📌 Project Overview

Low-light photographs often suffer from poor visibility, noise, and reduced contrast. This project implements an image enhancement pipeline that restores illumination using a supervised deep learning approach.

The model was trained on paired low-light and normal-light images from the **LOL v1** and **LOL v2** datasets.

Unlike traditional brightness adjustment techniques, the network learns to reconstruct well-lit images directly from low-light inputs.

---

## ✨ Features

- Deep Learning based image enhancement
- U-Net with pretrained ResNet34 encoder
- Trained on LOL v1 + LOL v2 datasets
- Hybrid loss function for improved restoration
- Streamlit Web Application
- CPU & GPU compatible
- Upload and download enhanced images

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Segmentation Models PyTorch (SMP)
- Torchvision
- Pillow
- NumPy
- Streamlit

---

# 📂 Dataset

The model was trained using:

- LOL Dataset v1
- LOL Dataset v2

Total paired training images:

**≈1100 images**

> The dataset is not included in this repository because of its large size.
https://www.kaggle.com/datasets/alenken/low-light-image-enhancement-datasets?utm_source=chatgpt.com
---

# 🧠 Model Architecture

Architecture:

```
Input Image
      │
      ▼
ResNet34 Encoder
      │
      ▼
      U-Net
      │
      ▼
Enhanced Image
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net |
| Library | Segmentation Models PyTorch |
| Encoder | ResNet34 |
| Encoder Weights | ImageNet |
| Input Channels | 3 |
| Output Channels | 3 |

---

# ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Image Size | 256 × 256 |

---

# 📉 Loss Function

Initially, the model was trained using **L1 Loss** only.

Although the model successfully enhanced brightness, the output images contained noticeable artifacts and noise.

To improve restoration quality, the training strategy was updated using **AI suggestions**.

Final loss consists of:

- L1 Loss
- SSIM Loss
- VGG Perceptual Loss

The model output is also clamped using:

```python
torch.clamp(output, 0, 1)
```

to ensure valid pixel values during inference.

---

# 🚀 Improvements

Compared to the initial model:

✅ Better brightness restoration

✅ Reduced image noise

✅ Improved structural similarity

✅ Cleaner output images

---

# 📸 Results in assets folder


## Model Comparison

The project evolved through multiple training iterations.

| Low Light | Initial Model (L1 Loss) | Final Model |
|------------|--------------------------|-------------|
| Image | Image | Image |

The final model produces cleaner and brighter images while reducing artifacts compared to the initial implementation.

---

# 💻 Streamlit Application

The web application allows users to:

- Upload a low-light image
- Enhance it using the trained model
- Compare original and enhanced images
- Download the enhanced image

---

# 📂 Project Structure

```
Low-Light-Image-Enhancement/
│
├── app.py
├── low_light_Unet.pth
├── requirements.txt
├── README.md
├── model.ipynb
├── assets/
│   ├── input.png
│   ├── output.png
│   ├── expected_output.png
│
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/JeevikaH22/Low_light_img_enhancement.git
```

Move to the project folder

```bash
cd Low-Light-Image-Enhancement
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 🔮 Future Improvements

- Train on larger datasets
- Improve edge sharpness
- Experiment with transformer-based architectures
- Train on better devices (better GPU and more epochs)
- Deploy using Docker
- Optimize inference speed

---

# 👩‍💻 Author

**Jeevika Hunnurkar**

Computer Engineering Student

Interested in:
- Machine Learning
- Deep Learning
- Computer Vision
- Data

---

## ⭐ Always ready to improve myself!!!