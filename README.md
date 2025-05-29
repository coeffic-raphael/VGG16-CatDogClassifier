# VGG16-CatDogClassifier

A deep learning project developed as part of an *Agentic AI* course.  
This model uses **transfer learning** with **VGG16** to classify images of cats and dogs with high accuracy.

## 🧠 Overview

- Architecture: VGG16 (pretrained on ImageNet)
- Training: 20,000+ labeled images (cats and dogs)
- Techniques: Fine-tuning, data augmentation, early stopping, learning rate scheduling
- Output: Trained `.keras` model and predictions exported to a CSV file

## 📂 Setup

1. **Download the dataset** from [Kaggle – Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)
2. Unzip the following two archives locally:
   - `train.zip` → contains 25,000 labeled images (cat/dog)
   - `test1.zip` → contains 12,500 unlabeled images for prediction
3. Place them in the following structure:
your_project_directory/
└── data/
├── train/
└── test1/

## 🏃 Running the project

- Open and run the provided notebook or script (e.g., in Google Colab or locally with GPU).
- The script will:
- Train the VGG16 model on the dataset
- Evaluate and display performance
- Save predictions to `submission.csv`
- Save the trained model to `model_vgg16.keras`

> ⚠️ The `.keras` file is too large to be uploaded to GitHub.  
> You must **run the code** yourself to generate the model file.

## 👨‍💻 Author

**Raphaël Coeffic** – Computer Science BSc student– AI enthusiast  
Contact: *racoeffic@gmail.com*

---

