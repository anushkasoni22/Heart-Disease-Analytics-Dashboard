<div align="center">

# 🫀 Heart Disease Analytics Dashboard
### A Data-Driven Approach to Saving Lives (CDC Data 2020 & 2022)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Desktop-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Analysis](https://img.shields.io/badge/Analysis-Predictive_%26_Descriptive-2874A6?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

[▶️ Watch Full Dashboard Demo (Google Drive Link)](https://drive.google.com/file/d/14qC33lMVLTstbZSffObQmyu6fRydWdtt/view?usp=sharing)
---

## 📖 Project Overview
This project is an end-to-end analytics dashboard designed to identify key drivers of heart disease and simulate the impact of lifestyle changes. Using a massive dataset from the **CDC (2020 & 2022)**, it moves beyond simple charts to provide **Predictive AI Insights** and **Prescriptive Strategies**. 

### 🎯 Key Objectives:
* **Automate Data Processing:** Build a robust Python/SQL backend to clean, merge, and impute over 500k+ records.
* **Analyze** the correlation between lifestyle choices (Smoking, Sleep, BMI) and Heart Health.
* **Identify** high-risk demographics and comorbidity hotspots.
* **Simulate** how reducing obesity and smoking rates can mathematically lower disease prevalence.

---

## 🧠 Backend Data Architecture & Machine Learning (Python/SQL)
To make the predictive models highly accurate and address real-world data anomalies, a custom automated backend pipeline was engineered before feeding the data into Power BI:

* **SQL In-Memory Database (`sqlite3`):** Avoided Disk I/O bottlenecks by loading raw CSVs into RAM. Executed `UNION ALL` SQL queries to align, rename, and merge the 2020 and 2022 datasets in milliseconds.
* **Optimized KNN Imputation:** Handled missing numerical data without triggering $O(N^2)$ memory crashes by training the `KNNImputer` on a representative sample before applying it to the entire 566k+ dataset.
* **Bias Mitigation (Sick Quitter Effect):** Corrected statistical illusions in the CDC data where severe patients quit drinking, falsely making alcohol appear protective. This was manually adjusted in the modeling phase to ensure the hospital-grade calculator does not provide dangerous medical recommendations.
* **Class-Balanced Probabilistic Scoring:** Addressed severe class imbalance (95% healthy, 5% diseased) using a calibrated **Logistic Regression** model with `class_weight='balanced'`. Used `predict_proba()` to generate a realistic and wide risk spread (1% to 99%) for the dynamic Power BI dashboard parameters.

---

## 📸 Dashboard Visual Walkthrough

### 🏠 Home: The Command Center
*An executive summary providing a centralized navigation hub and high-level key metrics.*

---

### 📄 Page 1: Descriptive Analysis (Demographics)
*Who is most at risk? A breakdown of age, gender, and perception.*
* **Visuals:** Impact of Smoking, Heart Risk by Sleep Duration, Risk by Age Group.
* **Key Insight:** analyzed the gap between "Health Perception vs. Reality".

---

### 🌍 Page 2: 2022 Geographic & Advanced Risk Factors
*A deep dive into the 2022 dataset focusing on environmental and specific physical traits.*
* **Hotspots:** US Geographic Risk Map.
* **Critical Factors:** Impact of **Lung Disease (COPD)**, **Mental Health**, and **Oral Health (Teeth Loss)** on heart risk.
* **Mobility:** Risk analysis based on Walking Difficulty.

---

### 🧬 Page 3: Deep Dive - Comorbidities & Lifestyle
*Understanding the "Lifestyle Paradox" and how conditions multiply.*
* **The Lifestyle Paradox:** Is "Sitting" worse than Smoking?
* **Root Cause Analysis:** How multiple conditions (Comorbidities) multiply the risk factor.
* **Warning Signs:** The impact of Chest Pain (Angina).

---

### ⚖️ Page 4: What Drives Heart Disease?
*Correlating physical attributes with disease probability.*
* **Metrics:** Average BMI Score & Smoker Ratios.
* **Visuals:** BMI vs. Physical Activity, Risk by Kidney Status, and Stroke History impact.

---

### 🤖 Page 5: Predictive Driver Analysis (AI Powered)
*Using Power BI's Key Influencers to find hidden patterns.*
* **Root Cause Analysis:** Automated insights identifying Top Factors (Age Category, Race, General Health) that increase risk.
* **Slicers:** Dynamic filtering by Age, Sex, and Smoking status.

---

### 🔮 Page 6: Prescriptive Analysis (Scenario Simulation)
*The most powerful feature: A "What-If" Simulation Engine.*
* **Simulation Engine:** Sliders to adjust **Physical Activity %**, **Smoking %**, and **Obesity %**.
* **Outcome:** A dynamic bar chart showing **"Current vs. Projected Impact"** and a counter for **"Lives Potentially Saved"**.
* **Strategy:** Recommendation Box with actionable health steps.

---

## 🛠️ Tech Stack & Workflow
| Category | Tools & Techniques Used |
| :--- | :--- |
| **Backend Data Pipeline** | Python (Pandas, `sqlite3`), SQL (In-Memory Database) |
| **Machine Learning** | Scikit-Learn (Logistic Regression, KNN Imputer, Probability Calibration) |
| **Tool** | Microsoft Power BI Desktop |
| **Data Processing** | Power Query (ETL), Data Cleaning, Merging 2020 & 2022 datasets |
| **Analysis** | DAX (Data Analysis Expressions), Measures, Calculated Columns |
| **Advanced** | Key Influencers AI Visual, What-If Parameters (Simulation) |
| **Design** | Custom Theme (#2C3E50), Navigation Buttons, Mobile Layout |

---

## 🔗 Data Source
* **Source:** [Kaggle: Indicators of Heart Disease (2022 Update)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
* *Note: This project is created for analytical and educational demonstration purposes.*

---
<div align="center">
⭐ <b>If you found this analysis insightful, please give this repo a star!</b> ⭐

</div>
