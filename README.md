# 🧠 Brain Tumor Classification using CNN & Transfer Learning

This project focuses on classifying brain tumor images into four categories using Convolutional Neural Networks (CNN) and Transfer Learning techniques. The main goal was to improve the accuracy of an existing model provided by our course instructor (which had ~76% accuracy).

---

## 🎯 Project Objective

- Analyze and improve the performance of the instructor-provided model.
- Apply different deep learning strategies to boost classification accuracy.
- Gain practical experience with CNNs, data preprocessing, augmentation, and transfer learning.

---

## 🚶 My Learning Journey

At the beginning of this project:
- I had **no prior hands-on experience** with building CNN models.
- I started by **watching YouTube tutorials** to understand how CNN and pretrained models like MobileNetV2 work.
- Then, with the help of **ChatGPT**, I iteratively refined and built my models step-by-step.

---

## ⚗️ Experiments Performed

I conducted **four major experiments**, each targeting improvement over the previous one:

---

### 🔬 Experiment 1: Basic CNN (Instructor Model)
- Used simple Conv2D + MaxPooling layers.
- No data augmentation or preprocessing.
- **Result**: Accuracy ~34% (much lower than expected).

---

### 🔬 Experiment 2: Added Data Augmentation
- Applied techniques like `RandomFlip`, `Rotation`, `Zoom` using `tf.keras.Sequential`.
- Helped reduce overfitting and increased feature generalization.
- **Result**: Accuracy improved to ~61%.

---

### 🔬 Experiment 3: Transfer Learning with MobileNetV2
- Loaded `MobileNetV2` with `imagenet` weights as the base model.
- Initially froze the layers and trained only the classifier head.
- Later fine-tuned the entire model with a reduced learning rate.
- Used `class_weight` to handle class imbalance.
- **Result**: Accuracy reached **71%**

---

### 🔬 Experiment 4: Final Model (No EarlyStopping)
- Removed `EarlyStopping` to allow full 30 epochs of training.
- The model continued to improve over more epochs.
- **Final Accuracy** reached: **76%** ✅
- Best overall class balance achieved in terms of precision and recall.

---

## ⚠️ Challenges Faced & How I Solved Them

| Challenge | Solution |
|----------|----------|
| Initial model gave very low accuracy (~34%) | Added data augmentation to increase diversity |
| Overfitting on training data | Used Dropout and augmented images |
| Class imbalance caused biased predictions | Applied `class_weight` during training |
| Low recall on specific class (pituitary tumor) | Fine-tuned pretrained model layers |
| EarlyStopping halted training too soon | Disabled it to let model train longer |
| Lack of deep understanding | Learned from YouTube and practiced with ChatGPT’s help |

---

## 🏆 Final Model Performance

| Class             | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Glioma Tumor     | 0.89      | 0.32   | 0.47     |
| Meningioma Tumor | 0.70      | 0.96   | 0.81     |
| No Tumor         | 0.73      | 1.00   | 0.84     |
| Pituitary Tumor  | 0.96      | 0.73   | 0.83     |

- ✅ **Overall Accuracy**: 76%  
- ✅ **Macro F1-score**: 74%  
- ✅ **Model Saved As**: `brain_tumor_classifier_v2_76acc.h5`

---

## 🧠 Technologies & Tools Used

- Python
- TensorFlow / Keras
- Google Colab
- MobileNetV2 (Pretrained Model)
- Matplotlib, Seaborn for Visualization
- scikit-learn for evaluation metrics

---

## 💾 Model Files

- `brain_tumor_classifier_v2_76acc.h5` – Final model with 76% accuracy
- `classification_report.txt` – Precision, Recall, F1-score summary
- `confusion_matrix.png` – Visual overview of predictions

---

## 🙏 Acknowledgements

- Special thanks to our course instructor for providing the initial model and dataset.
- YouTube content creators for beginner-friendly tutorials.
- ChatGPT (by OpenAI) for guidance throughout model development, debugging, and optimization.

---

## 📌 Summary

This project taught me how to go from **zero understanding** to building and optimizing deep learning models for real-world medical image classification. I gained confidence in experimenting, analyzing results, and tuning models based on proper evaluation.

