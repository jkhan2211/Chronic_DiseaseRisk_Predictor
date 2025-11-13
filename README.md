# Symptom-Based Diagnostic Decision Support AI System (DDSAS)

A machine learning project predicting the likelihood of chronic diseases based on patient symptoms and health metrics. This end-to-end system demonstrates MLOps principles, with data preprocessing, model training, MLflow experiment tracking, and a local Streamlit UI demo.  

It provides a **data-driven decision support tool** for healthcare settings.

---

## 🎯 Project Overview 

The stakeholders of DDSAS include:

- **Physicians & Healthcare Providers:** Aid in disease diagnosis, improve accuracy, and reduce errors.  
- **Patients:** Benefit from timely and accurate diagnosis for better health outcomes.  
- **Healthcare Institutions & Clinics:** Improve workflows, standardize care, and reduce operational costs.  
- **Health IT & Data Scientists:** Build, track, and maintain ML models efficiently.  
- **Regulatory Bodies:** Ensure compliance with safety, ethics, and medical standards.  
- **Insurance Companies:** Optimize cost-effectiveness by minimizing misdiagnosis.  
- **Medical Educators & Researchers:** Train professionals and advance diagnostic methods.  
- **Patients’ Families & Caregivers:** Indirectly benefit from improved care decisions.  

This project helps all stakeholders make **informed, timely decisions** in healthcare.

## Clinical Risks

It is important to note that DDSAS is designed as a diagnostic support tool intended to assist front-line healthcare professionals. Patient well-being is central to our approach; the system is not meant to deliver definitive diagnoses based solely on symptom patterns. Instead, it aims to support nurses, physicians, and telehealth providers by offering data-driven predictions that complement their clinical expertise.

Rather than presenting a conclusive prognosis, the tool provides probabilistic estimates of potential diseases based on reported symptoms, helping guide clinicians toward more informed decisions and appropriate next steps in patient care.


## 🧩 Folder Structure

```
chronic_disease_risk_predictor
├── data #Raw Dataset
│   ├── Disease_Prediction.csv
├── src
│   ├── streamlit_app.py # Interactive demo UI
├── requirements.txt
└── README.md # Project overview
```
---
## Dataset
The dataset selected for this task is “Disease Prediction Using Machine Learning” (Kaggle link). It was chosen because it includes a large number of symptoms (features) and corresponding prognoses (target classes). Given its size and structure, the dataset is well-suited for this project, as it reflects the scale of real-world healthcare data, where hospitals manage large patient populations and numerous clinical variables. This allows us to test whether our model can effectively handle datasets of comparable complexity.

The dataset is already divided into training and testing subsets. In total, it contains data from 4,962 individuals with 133 possible symptoms and 42 diagnosed diseases. However, there is no accompanying metadata, so additional information such as patient demographics or age distributions cannot be analyzed.



## 🤝 Team Members

[Junaid Khan](https://www.linkedin.com/in/junaid-devops)• 
[Adam Healey](https://www.linkedin.com/in/adam-healey/) • 
[Ali Hyder]() • 
[Olga Nazarenko]() • 
[Pradeep Venkatesan]()

---

## 📦 Technologies Used

| Component           | Technology       | Purpose                        |
|--------------------|----------------|--------------------------------|
| Data Preprocessing       | pandas, numpy   | Clean & prepare dataset        |
| Visualization         | Matplotlib, seaborn, plotly | Visual data summaries  |
|Exploratory Data Analysis | DBScan, PCA    | Dimensionality reduction and clustering | 
| Machine Learning    | scikit-learn, xgboost | Train predictive models   |
| Experiment Tracking | MLflow          | Log experiments & model metrics|
| UI / Demo           | Streamlit       | Local interactive interface   |

---

## Sample Classification Models to Try

| Model                  | Description                                   | Assigned To         |
|------------------------|-----------------------------------------------|------------------|
| Logistic Regression    | Baseline probabilistic classifier             |    AH  |
| Random Forest          | Ensemble of decision trees, robust to overfitting |        ALH|
| XGBoost                | Gradient boosting, effective on tabular data | ON        |
| LightGBM               | Fast gradient boosting, handles large data   |   JK |
| SVM                    | Good for high-dimensional, complex boundaries | PV |
| Neural Networks (MLP)  | Deep learning for complex feature interactions |       |

---

## Sample Clustering Models to Try

| Model                  | Description                                   | Assigned To       |
|------------------------|-----------------------------------------------|----------------|
| KMeans                 | Partition-based clustering                     |   ON  |
| DBSCAN                 | Density-based, finds arbitrarily shaped clusters |      ALH|
| Agglomerative          | Hierarchical clustering                        | PV |

## Pre-processing
It is important to note that while the open-source dataset has already been split into a training and testing components, inspection of the sizes of these dataframes shows that the testing dataset represents only 1% of the training set, which according to data science and model training best practices is inadequate.  To remedy this issue, we opted to recombine the training and testing dataset to create new training/test dataframes using a 80/20% split of patients.
```
# Basic library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the training data
df = pd.read_csv("/Users/adamhealey/Downloads/Training.csv")
print("Dataset shape:", df.shape)
df.head()

# 2. Load the testing data
df2 = pd.read_csv("/Users/adamhealey/Downloads/Testing.csv")
print("Dataset shape:", df2.shape)
df2.head()
```
```
#look at column names and look to see which is different between the test and training set

#save the column names of each dataframe to a set
colSet1 = set(df.columns.tolist())
colSet2 = set(df2.columns.tolist())

#use set intersection to find which columns are not in both datasets
diff = colSet1 - colSet2
print(diff)
```
The training dataset has an extra column called 'Unnamed: 133' that is not in the test set.  This column should be removed before merging

```
df = df.drop(columns=['Unnamed: 133'])

#re-check to ensure both dataframes have the same columns
colSet1 = set(df.columns.tolist())
colSet2 = set(df2.columns.tolist())
diff = colSet1 - colSet2
print(diff)

#check reverse as well:
diff2 = colSet1 - colSet2
print(diff2)
```
Columns match.  Dataframes are ready to be merged.
```
#Use pandas concatenate to combine rows of each dataset.
#also, check first to make sure the columns are in the same order in both datasets.

mergedDF = pd.concat([df, df2[df.columns]], # this ensures the correct ordering by calling df colummns first
                     ignore_index=True)
mergedDF.shape
```
INSERT ADDITIONAL PREPROCESSING HERE.

## Exploratory Data Analysis (EDA)
Now that we have our newly combined dataset, exploratory analysis can begin.

Questions to explore in the dataset:
1. How clean is the dataset?  Are there missing data or symptoms that are never reported?

2. What is the distribution of symptoms among individuals? 

3. How many symptoms in total are present? How do symptoms cluster with one another?

4. Are there individuals with identical symptom patterns?
```
#Question 1- how clean is the dataset?
#first, the prognosis column should be removed as it is a predictor, not a feature
featuresDF = mergedDF.drop(columns=['prognosis'])
# are there NAs?
na_cols = featuresDF.columns[featuresDF.isna().any()] #get the names of the columns containing NAs
print("Columns with NaNs:", na_cols.tolist())

#checking for symptoms that are never observed among individuals
zero_sum_cols = featuresDF.columns[featuresDF.sum() == 0]

print("Columns with sum = 0:", list(zero_sum_cols))
```
"Fluid_overload" is never observed as a symptom among patients.  Prior to building a model, this syptom should be removed.

There are no NA values in the dataset.

```
#Question 2: what is the distribution of symptoms among individuals?
#This is a binary matrix with no missing values so you can simply count the number of 1s per row and calculate the mean
avgSymptoms = featuresDF.sum(axis=1).mean()
avgSymptoms
```
The average number of symptoms per patient is: 7.4.  
Next, use a histogram to inspect the distribution of symptoms among individuals.

```
plt.figure(figsize=(8,8))
plt.grid(False)
plt.hist(featuresDF.sum(axis=1), 
         bins = 15)
plt.title("Histogram distribution of Symptoms per individual")
plt.xlabel("Number of symptoms")
plt.ylabel("Symptom Counts")
```
## 📦 Demo

Video Link:
---


## ⚙️ Setup & Usage

1. **Clone the repository**
```bash
git clone https://github.com/<username>/Chronic_DiseaseRisk_Predictor.git
cd Chronic_DiseaseRisk_Predictor

2. **Clone the repository**

```bash
pip install -r requirements.txt


