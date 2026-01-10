Here is the **GitHub README.md** based on your project details, architecture, and our debugging process. It highlights the "From Scratch" nature of the work and focuses on the results and inference.

---

# Custom Faster R-CNN: Object Detection from Scratch

A complete, custom implementation of the **Faster R-CNN** object detection architecture, built and trained entirely from scratch (without pre-trained ImageNet weights) on the PASCAL VOC dataset.

### 🎥 Real-Time Detection Demo

*(Note: Replace `results/output_detection.gif` with your actual GIF path)*

---

## 📖 Overview

This project demonstrates the construction of a two-stage object detector using **PyTorch**. Unlike standard transfer learning approaches, this model initializes the **VGG16 backbone** with random weights, requiring a specialized training regimen to achieve convergence.

**Key Features:**

* **Zero Pre-training:** Random initialization of all layers (Backbone, RPN, Head).
* **Custom Pipeline:** End-to-end implementation including XML parsing, Anchor generation, and RoI Pooling.
* **Video Inference:** Optimized script for frame-by-frame detection on video streams.

---

## 🧠 Model Architecture

The architecture is a classic Faster R-CNN design adapted for stability during "from-scratch" training.

| Component | Specification |
| --- | --- |
| **Backbone** | **VGG16** (features layers only). `pretrained=False`. |
| **RPN** | **Region Proposal Network** with 9 anchors (3 scales, 3 aspect ratios). |
| **RoI Head** | **RoIPool** (7x7) → 2x FC Layers (1024 dim) → Classification/Regression. |
| **Input Size** | Dynamic resizing (Min: 600px, Max: 1000px). |
| **Classes** | 20 Object Classes (PASCAL VOC) + 1 Background. |

---

## 🔬 Methodology

To successfully train deep object detection networks without pre-trained weights, specific strategies were employed:

* **Data Augmentation:** Random Horizontal Flipping to artificially double the dataset size and introduce spatial variance.
* **Gradient Accumulation:** Implemented an accumulation step of 4 to simulate larger batch sizes, stabilizing the noisy gradients typical of scratch training.
* **Learning Rate Schedule:** A multi-step decay strategy (Steps at Epoch 35 & 45) was used over **50 Epochs** to refine feature extraction after the initial "coarse" learning phase.
* **Class Handling:** Strict alphabetical sorting of PASCAL VOC classes to ensure consistent label mapping between training and inference.

---

## 📊 Performance Metrics

*Evaluation performed on PASCAL VOC Validation Set.*

| Metric | Result |
| --- | --- |
| **mAP (Mean Average Precision)** | **[INSERT mAP HERE]** |
| **Inference Speed** | **16.5** FPS (on GPU) |
| **Model Size** | **167.64** MB |

> **Trade-off Analysis:** While VGG16 provides robust feature extraction, its parameter density (~138M params) limits real-time performance compared to lighter backbones like ResNet-50. The "from scratch" constraint also necessitates longer training times (50+ epochs) compared to fine-tuning (10-20 epochs).

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/custom-faster-rcnn.git
cd custom-faster-rcnn

```


2. **Install Dependencies:**
```bash
pip install torch torchvision opencv-python pyyaml tqdm pillow numpy

```


3. **Prepare Weights:**
Ensure your trained model file (e.g., `faster_rcnn_voc_scratch.pth`) is placed in the `voc_scratch_kaggle/` directory (or update the config path accordingly).

---

## 🚀 Usage

### 1. Run Evaluation (mAP)

To calculate the Mean Average Precision on the test set:

```bash
python tools/infer.py --config config/voc.yaml

```

### 2. Video Inference

To run the object detector on a video file:

```bash
python tools/video_demo.py \
    --config config/voc.yaml \
    --video_path path/to/your/video.mp4 \
    --threshold 0.7

```

* **`--threshold`**: Confidence threshold (default: 0.7). Lower this to 0.3-0.5 if detections are sparse.
* **Output**: The processed video will be saved as `output_demo.avi`.

---

## 📂 Dataset

This model was trained on the **PASCAL VOC 2012** dataset.

* **Training:** Train/Val split.
* **Classes:** Person, Bird, Cat, Cow, Dog, Horse, Sheep, Aeroplane, Bicycle, Boat, Bus, Car, Motorbike, Train, Bottle, Chair, Dining Table, Potted Plant, Sofa, TV/Monitor.

---

## 📜 Acknowledgments

* Original Faster R-CNN Paper: [Ren et al., 2015](https://arxiv.org/abs/1506.01497)
* PASCAL VOC Dataset Team.
