# 🛒 Bangladesh E-Commerce Reviews

## Statistical Analysis • NLP • Sentiment Analysis • Emotion Classification • Machine Learning

> **From raw customer reviews to statistical insight and machine learning predictions.**

This project explores a large-scale Bangladesh e-commerce review dataset through a complete **Data Science + Machine Learning workflow**.

The goal is not simply to train a model.

The goal is to understand the data **before** modeling, investigate meaningful relationships statistically, transform unstructured customer reviews into machine-readable features, and finally build machine learning models capable of predicting customer **sentiment** and **emotion** from review text.

---

## 🎯 Project Objective

Customer reviews contain valuable information about how people experience products and services.

But raw reviews are unstructured text.

This project asks:

```text
What can we learn from customer reviews?
              ↓
How is customer sentiment distributed?
              ↓
What emotions appear most frequently?
              ↓
How do ratings relate to sentiment?
              ↓
Do product categories show different sentiment patterns?
              ↓
Can statistical relationships be identified?
              ↓
Can machine learning learn sentiment from review text?
              ↓
Can machine learning identify the emotion expressed in a review?
```

The project therefore combines:

**Exploratory Data Analysis + Statistics + NLP + Machine Learning**

---

# 📊 Dataset

The dataset is based on Bangladesh e-commerce customer reviews collected from:

* 🛍️ Daraz
* 🛍️ Pickaboo

The dataset contains **78,130 reviews**, including:

| Language  |    Reviews |
| --------- | ---------: |
| Bangla    |     28,912 |
| English   |     49,218 |
| **Total** | **78,130** |

Each observation represents a customer review.

### Dataset Features

```text
Rating
Review
Product Name
Product Category
Emotion
Sentiment
Data Source
```

The dataset provides sentiment and emotion labels that can be used as supervised learning targets.

---

# 🧠 The Core Data Science Question

A major distinction in this project is:

```text
Statistical Analysis
        ↓
Understand the data

Machine Learning
        ↓
Learn patterns from the data
        ↓
Make predictions
```

The dataset already contains `Sentiment` and `Emotion` labels.

Therefore, during supervised learning:

```text
                 X
                 ↓
            Review Text
                 ↓
          Feature Engineering
                 ↓
            ML Algorithm
                 ↓
                 y
        Sentiment / Emotion
```

We do **not** give the target label to the model as an input.

---

# 🔬 Project Architecture

```text
                         RAW DATA
                            │
                            ▼
                  ┌──────────────────┐
                  │ Data Understanding│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Data Cleaning    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ EDA              │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Statistical       │
                  │ Analysis         │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ NLP Preprocessing│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Feature          │
                  │ Engineering      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Machine Learning │
                  └────────┬─────────┘
                           │
                  ┌────────┴─────────┐
                  ▼                  ▼
           Sentiment Model     Emotion Model
                  │                  │
                  ▼                  ▼
          Positive/Negative    5 Emotion Classes
```

---

# 📈 Phase 1: Statistical Analysis

Before building machine learning models, we investigate the structure of the dataset.

## 1. Data Quality Analysis

We examine:

* Missing values
* Duplicate observations
* Data types
* Unique values
* Invalid values
* Inconsistent categories
* Text quality
* Class distributions

---

## 2. Descriptive Statistics

We investigate the basic statistical structure of the data.

Examples:

```text
Mean
Median
Mode
Minimum
Maximum
Frequency
Proportion
Distribution
```

For categorical variables, we focus particularly on:

```text
Frequency
Percentage
Class Distribution
```

---

# 💬 Sentiment Distribution

The dataset contains two sentiment classes:

```text
Positive
Negative
```

The documented dataset distribution is approximately:

```text
Positive   █████████████████████████████████████ 86.1%
Negative   ██████                                13.9%
```

This immediately raises an important machine learning issue:

> ⚠️ **Class imbalance**

A model that achieves high accuracy may still perform poorly on the minority class.

Therefore, model evaluation will not rely on accuracy alone.

---

# ❤️ Emotion Distribution

The dataset contains five emotion categories:

```text
Happiness
Love
Sadness
Anger
Fear
```

Distribution:

```text
Happiness   46,635
Love        20,633
Sadness      6,533
Anger        3,171
Fear         1,158
```

Visualizing these classes helps us understand whether the emotion prediction problem is balanced or dominated by particular classes.

---

# 📊 Statistical Questions

Instead of creating visualizations only for appearance, this project uses them to investigate actual questions.

### Rating vs Sentiment

```text
Rating
  │
  ├── 1 ⭐
  ├── 2 ⭐
  ├── 3 ⭐
  ├── 4 ⭐
  └── 5 ⭐
       │
       ▼
   Sentiment
```

**Question:**

> Is customer rating associated with sentiment?

---

### Product Category vs Sentiment

```text
Product Category
       │
       ├── Electronics
       ├── Clothing
       ├── Watches
       ├── Tools
       └── ...
              │
              ▼
          Sentiment
```

**Question:**

> Does sentiment distribution differ across product categories?

---

### Data Source vs Sentiment

```text
Data Source
     │
 ┌───┴────┐
 ▼        ▼
Daraz   Pickaboo
 │        │
 └───┬────┘
     ▼
 Sentiment
```

**Question:**

> Is sentiment distribution different between the two e-commerce platforms?

---

# 🧪 Statistical Testing

Where appropriate, relationships observed during EDA will be formally investigated using statistical tests.

The workflow is:

```text
Research Question
       ↓
Define Variables
       ↓
Formulate Hypotheses
       ↓
Choose Statistical Test
       ↓
Calculate Test Statistic
       ↓
Calculate p-value
       ↓
Interpret Result
```

The important principle is:

> **Visualization shows a pattern. Statistical testing investigates whether the observed relationship is supported by evidence under the chosen assumptions.**

---

# 🧹 Phase 2: NLP

Customer reviews are text.

Machine learning algorithms cannot directly understand raw sentences in their original form.

Therefore:

```text
Raw Review
    ↓
Text Cleaning
    ↓
Normalization
    ↓
Tokenization / Representation
    ↓
Numerical Features
    ↓
Machine Learning
```

The exact preprocessing pipeline will depend on the characteristics of the review text and the modeling approach.

---

# 🤖 Phase 3: Machine Learning

The project contains two natural supervised learning tasks.

## 🎯 Task 1: Sentiment Classification

### Input

```text
Review
```

### Target

```text
Sentiment
```

### Prediction

```text
Positive
Negative
```

Conceptually:

```text
"I really love this product"
             ↓
        NLP Pipeline
             ↓
       ML Classifier
             ↓
         Positive
```

---

# ❤️ Task 2: Emotion Classification

### Input

```text
Review
```

### Target

```text
Emotion
```

### Prediction

```text
Happiness
Love
Sadness
Anger
Fear
```

Conceptually:

```text
"I am extremely happy with this product"
                    ↓
              NLP Pipeline
                    ↓
              ML Classifier
                    ↓
                Happiness
```

---

# ⚠️ Avoiding Data Leakage

One of the most important principles in this project is avoiding **data leakage**.

Suppose:

```text
Review = "Amazing product!"
Rating = 5
Sentiment = Positive
```

If our goal is:

```text
Review → Sentiment
```

then giving the model:

```text
Review + Rating + Sentiment
```

would allow the model to directly access information related to the target.

The intended setup is:

```text
X = Review

        ↓

     ML Model

        ↓

y = Sentiment
```

Similarly:

```text
X = Review

        ↓

     ML Model

        ↓

y = Emotion
```

This keeps the prediction problem clearly defined.

---

# 📏 Model Evaluation

Because the dataset contains imbalanced classes, model evaluation will consider multiple metrics rather than accuracy alone.

Potential evaluation metrics include:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

For multi-class emotion classification, class-level performance will be especially important.

---

# 🔍 What This Project Is Really Demonstrating

This repository is designed to demonstrate a complete Data Science workflow:

```text
                 DATA
                  │
                  ▼
          Understand the Data
                  │
                  ▼
            Clean the Data
                  │
                  ▼
          Explore the Data
                  │
                  ▼
        Ask Statistical Questions
                  │
                  ▼
        Test Statistical Evidence
                  │
                  ▼
       Understand Textual Data
                  │
                  ▼
          Build NLP Features
                  │
                  ▼
          Train ML Models
                  │
                  ▼
        Evaluate Predictions
                  │
                  ▼
          Interpret Results
```

The important idea is:

> **Machine Learning is not the first step. Understanding the data is.**

---

# 🗂️ Repository Structure

```text
bangladesh-ecommerce-reviews/
│
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_nlp_preprocessing.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_sentiment_model.ipynb
│   └── 08_emotion_model.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── statistics/
│   ├── features/
│   └── models/
│
├── reports/
│   └── figures/
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technology Stack

### Programming

```text
Python
```

### Data Analysis

```text
NumPy
Pandas
```

### Visualization

```text
Matplotlib
Seaborn
```

### Statistics

```text
SciPy
Statistical Testing
Probability
Descriptive Statistics
```

### Machine Learning

```text
Scikit-learn
```

### NLP

```text
Text preprocessing
Feature extraction
Text classification
```

---

# 🚀 Learning Goals

This project is being developed as a practical learning project to connect theoretical concepts with a real dataset.

The main learning objectives are:

* Understand real-world data
* Perform systematic EDA
* Apply descriptive statistics
* Formulate statistical questions
* Perform hypothesis testing
* Understand class imbalance
* Understand data leakage
* Work with unstructured text
* Perform NLP preprocessing
* Convert text into numerical representations
* Build classification models
* Evaluate models using multiple metrics
* Interpret statistical and ML results
* Connect statistics with machine learning

---

# 🧩 The Big Picture

```text
             CUSTOMER REVIEWS
                    │
                    ▼
          ┌──────────────────┐
          │   DATA ANALYSIS  │
          └────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   Statistics               NLP
        │                     │
        ▼                     ▼
 Relationships          Text Features
 Distributions                │
 Hypothesis Tests             ▼
        │                Machine Learning
        │                     │
        └──────────┬──────────┘
                   ▼
              INSIGHTS
                   +
             PREDICTIONS
```

---

# 🌱 Future Work

Possible extensions include:

* Comparing multiple classification algorithms
* Hyperparameter tuning
* Cross-validation
* Advanced NLP representations
* Bangla-specific NLP approaches
* Multilingual modeling
* Error analysis
* Model interpretability
* More detailed statistical investigation
* Deployment of the final model

---

# 📚 Dataset Source

The project is based on the research dataset:

**“A comprehensive dataset for sentiment and emotion classification from Bangladesh e-commerce reviews.”**

The dataset article describes the collection and annotation of Bangladesh e-commerce reviews for sentiment and emotion classification and identifies applications including machine learning, NLP, statistical analysis, and data mining.

---

# 👩‍💻 Project Philosophy

This project follows one principle:

> ### **Don't just train a model. Understand the data that the model is learning from.**

Statistics helps us understand the data.

NLP helps us represent language.

Machine Learning helps us learn patterns.

Together, they form a complete Data Science workflow.

---

## ⭐ Project Status

```text
🟢 Dataset Understanding
🟢 Project Definition
🟡 Statistical Analysis
⚪ NLP Pipeline
⚪ Feature Engineering
⚪ Sentiment Model
⚪ Emotion Model
⚪ Model Evaluation
⚪ Final Analysis
```

**Status: 🚧 In Progress**

---

## 🔥 Final Pipeline

```text
RAW REVIEWS
     │
     ▼
DATA UNDERSTANDING
     │
     ▼
DATA CLEANING
     │
     ▼
EDA
     │
     ▼
STATISTICAL ANALYSIS
     │
     ▼
NLP PREPROCESSING
     │
     ▼
FEATURE ENGINEERING
     │
     ▼
MACHINE LEARNING
     │
     ├───────────────┐
     ▼               ▼
SENTIMENT         EMOTION
CLASSIFICATION    CLASSIFICATION
     │               │
     └───────┬───────┘
             ▼
      MODEL EVALUATION
             │
             ▼
      INTERPRETATION
             │
             ▼
       FINAL INSIGHTS
```

**A complete journey from customer language → statistical evidence → machine learning prediction.** 🚀
