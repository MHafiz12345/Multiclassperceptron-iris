# 🌸 Multi-Class Perceptron from Scratch on Iris Dataset

This project builds a **multi-class Perceptron classifier** from scratch to classify Iris flowers into three species. The implementation includes manual preprocessing, training logic, evaluation, and an automated experimental framework. No external machine learning libraries are used for the modeling process.

## 📂 Dataset

The dataset used is the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), included in the repository as `irisdataset.csv`.

Each sample contains:
- **Features**: `sepal.length`, `sepal.width`, `petal.length`, `petal.width`
- **Target**: `variety` (species class: *setosa*, *versicolor*, *virginica*)

## ⚙️ Manual Data Preprocessing

Data preprocessing is handled manually, without pandas' built-in transformations or scikit-learn.

### 🔸 Steps:

1. **Missing Values & Duplicates**:
   - Dataset is checked for nulls and duplicates
   - Duplicates are removed

2. **Label Encoding**:
   - Classes are manually encoded to integers:
     - `setosa → 0`
     - `versicolor → 1`
     - `virginica → 2`

3. **Z-score Standardization**:
   - All numeric features are standardized:
   
   ![z = \frac{x - \mu}{\sigma}](https://latex.codecogs.com/svg.latex?z%20%3D%20%5Cfrac%7Bx%20-%20%5Cmu%7D%7B%5Csigma%7D)

4. **Final Dataset Saved**:
   - Output file: `iris_processed_standardized.csv`
   - Includes encoded class (`target`) and original label (`variety`)

## 🧠 Multi-Class Perceptron Classifier

A Perceptron is a linear classifier that updates its weights whenever a misclassification occurs. This implementation supports multi-class classification using the **One-vs-All strategy** internally with a unified weight matrix.

### 🔹 Key Features:
- Supports **random or zero initialization**
- Uses **argmax over raw scores** for prediction
- Stops training on convergence or after 1000 epochs

## 🚀 Training Process

Training is conducted using this formula:

```
weights_true += η · x
weights_predicted -= η · x
```

Where:
- η: learning rate
- x: feature vector (with bias)
- The weights are only updated when a misclassification occurs

Each epoch logs:
- Misclassified samples
- Accuracy

## 📊 Evaluation Metrics

Model evaluation includes:
- **Accuracy**
- **Confusion Matrix**

The confusion matrix compares predictions across all 3 classes:

```
                Pred setosa  Pred versicolor  Pred virginica
True setosa         9              0               0
True versicolor     0              12              1
True virginica      0              2               11
```

## 🧪 Experimentation Framework

An automated script tests different model configurations by varying:
- 📏 Train-test split: `80/20`, `70/30`, `60/40`
- 🔁 Learning rates: `0.01`, `0.1`
- 🎲 Weight initialization: Zero vs. Random

Each configuration is evaluated for:
- Epochs to converge
- Test accuracy
- Training behavior

A summary table shows all results, followed by:
- Split-wise accuracy and training trends
- Effect of learning rate
- Effect of weight initialization
- Best configuration based on accuracy

## 🛠️ How to Run

1. Download the repo and open in Google Colab or a local environment
2. Run the preprocessing notebook to generate the processed dataset
3. Run the training and experimentation notebook

Notebooks:
- `iris_data_preprocessing.ipynb`
- `multiclass_perceptron_iris.ipynb`

## ✅ Output Example

```
Configuration:
Train-Test Split: 80%-20%
Learning Rate: 0.1
Weight Initialization: Random

Training completed in 9 epochs
Confusion Matrix: ...
Accuracy: 96.67%
Test Accuracy: 0.9667
```
