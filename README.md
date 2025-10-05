# Plant Disease Classification using CNN

This project focuses on building a Convolutional Neural Network (CNN) model to classify plant diseases using the **PlantVillage Dataset**. The dataset contains images of healthy and diseased plant leaves across **38 different classes**. The notebook demonstrates the end-to-end pipeline including dataset download, preprocessing, visualization, model training, and evaluation.

---

## 📂 Dataset

* Source: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* Size: ~2 GB
* Classes: 38 (e.g., Tomato___Bacterial_spot, Apple___Cedar_apple_rust, Grape___healthy, etc.)
* Data formats available:

  * **Color**
  * **Grayscale**
  * **Segmented**

---

## ⚙️ Features of the Notebook

1. **Reproducibility**
   Sets seeds for Python, NumPy, and TensorFlow to ensure consistent results.

2. **Data Curation**

   * Downloads dataset via Kaggle API.
   * Extracts and organizes images.
   * Confirms number of classes and samples.

3. **Preprocessing**

   * Resizes images to `224x224`.
   * Normalizes pixel values (`rescale=1./255`).
   * Uses `ImageDataGenerator` for **train-validation split**.

4. **Model Training** *(to be added later in the notebook)*

   * CNN-based architecture using TensorFlow/Keras.
   * Trained with `categorical_crossentropy` loss and `Adam` optimizer.

5. **Visualization**

   * Displays sample images.
   * Outputs shape details and pixel-level data.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab** (GPU recommended).
2. Upload your `kaggle.json` credentials (download from [Kaggle account settings](https://www.kaggle.com/account)).
3. Run the notebook cells step by step:

   * Install Kaggle API.
   * Download and unzip dataset.
   * Preprocess data and generate train/test splits.
   * Train and evaluate the CNN model.

---

## 🛠️ Dependencies

* Python 3.x
* NumPy
* Matplotlib
* TensorFlow / Keras
* PIL (Pillow)
* Kaggle API

Install requirements:

```bash
pip install kaggle tensorflow matplotlib pillow
```

---

## 📊 Results

* Number of classes: **38**
* Batch size: **32**
* Input image size: **224x224**

Expected outcomes:

* Trained CNN that can classify plant leaves into their correct disease/health category.
* Accuracy and loss plots to monitor training.

---

## 🔮 Future Improvements

* Use **transfer learning** (e.g., VGG16, ResNet50, EfficientNet).
* Add **data augmentation** for better generalization.
* Deploy model as a web app for farmers/agriculture experts.

---

## 📌 License

This project is intended for **educational and research purposes only**. Dataset is credited to PlantVillage and Kaggle contributors.
