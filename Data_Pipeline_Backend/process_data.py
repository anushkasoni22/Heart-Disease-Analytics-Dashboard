import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer

def main():
    print("=== Step 1: Ingesting raw CSV files into SQLite (In-Memory) ===")
    # Connect to an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    
    # Load raw CSVs
    csv_2020 = 'heart_2020_cleaned.csv'
    csv_2022 = 'heart_2022_no_nans.csv'
    
    if not os.path.exists(csv_2020) or not os.path.exists(csv_2022):
        print(f"Error: Raw CSV files '{csv_2020}' or '{csv_2022}' not found in the current directory.")
        return
        
    print(f"Loading '{csv_2020}' into table 'heart_2020'...")
    df_2020_raw = pd.read_csv(csv_2020)
    df_2020_raw.to_sql('heart_2020', conn, index=False, if_exists='replace')
    
    print(f"Loading '{csv_2022}' into table 'heart_2022'...")
    df_2022_raw = pd.read_csv(csv_2022)
    df_2022_raw.to_sql('heart_2022', conn, index=False, if_exists='replace')
    
    print("Ingestion complete.\n")
    
    print("=== Step 2: Executing SQL query to combine and align datasets ===")
    # Write SQL query to perform UNION ALL, column renaming, and column filtering
    sql_query = """
    SELECT 
        HeartDisease,
        BMI,
        Smoking,
        AlcoholDrinking,
        Stroke,
        PhysicalHealth,
        MentalHealth,
        DiffWalking AS DifficultyWalking,
        Sex,
        AgeCategory,
        Race,
        Diabetic,
        PhysicalActivity,
        GenHealth AS GeneralHealth,
        SleepTime,
        Asthma,
        KidneyDisease,
        SkinCancer,
        2020 AS Year
    FROM heart_2020
    
    UNION ALL
    
    SELECT 
        HadHeartAttack AS HeartDisease,
        BMI,
        SmokerStatus AS Smoking,
        AlcoholDrinkers AS AlcoholDrinking,
        HadStroke AS Stroke,
        PhysicalHealthDays AS PhysicalHealth,
        MentalHealthDays AS MentalHealth,
        DifficultyWalking,
        Sex,
        AgeCategory,
        RaceEthnicityCategory AS Race,
        HadDiabetes AS Diabetic,
        PhysicalActivities AS PhysicalActivity,
        GeneralHealth,
        SleepHours AS SleepTime,
        HadAsthma AS Asthma,
        HadKidneyDisease AS KidneyDisease,
        HadSkinCancer AS SkinCancer,
        2022 AS Year
    FROM heart_2022
    """
    
    df_merged = pd.read_sql_query(sql_query, conn)
    conn.close()
    print(f"Merged dataset shape: {df_merged.shape}")
    print("SQL execution complete.\n")
    
    print("=== Step 3: Categorical Normalization and Cleaning in Pandas ===")
    # Trim leading and trailing spaces from all object columns
    for col in df_merged.select_dtypes(include=['object']).columns:
        df_merged[col] = df_merged[col].astype(str).str.strip()
        
    # Standardize Smoking for 2022 rows
    smoking_map = {
        'Current smoker - now smokes every day': 'Yes',
        'Current smoker - now smokes some days': 'Yes',
        'Former smoker': 'No',
        'Never smoked': 'No'
    }
    # Note: 2020 is already Yes/No, so this will only affect 2022 long categories
    df_merged['Smoking'] = df_merged['Smoking'].replace(smoking_map)
    
    # Standardize AgeCategory (e.g. "Age 65 to 69" -> "65-69")
    df_merged['AgeCategory'] = df_merged['AgeCategory'].str.replace('Age ', '', regex=False)
    df_merged['AgeCategory'] = df_merged['AgeCategory'].str.replace('to', '-', regex=False)
    df_merged['AgeCategory'] = df_merged['AgeCategory'].str.replace(' - ', '-', regex=False)
    df_merged['AgeCategory'] = df_merged['AgeCategory'].str.strip()
    
    # Standardize Race (e.g. "White only, Non-Hispanic" -> "White")
    df_merged['Race'] = df_merged['Race'].str.replace('only, Non-Hispanic', '', regex=False)
    df_merged['Race'] = df_merged['Race'].str.replace('Multiracial, Non-Hispanic', 'Other', regex=False)
    df_merged['Race'] = df_merged['Race'].str.strip()
    # Align 'Other race' and 'Other' to 'Other'
    df_merged['Race'] = df_merged['Race'].replace({'Other race': 'Other'})
    
    # Standardize Diabetic (e.g. "No, pre-diabetes or borderline diabetes" -> "No, borderline diabetes")
    diabetic_map = {
        'No, pre-diabetes or borderline diabetes': 'No, borderline diabetes',
        'Yes, but only during pregnancy (female)': 'Yes (during pregnancy)'
    }
    df_merged['Diabetic'] = df_merged['Diabetic'].replace(diabetic_map)
    
    # Check category values to ensure alignment
    print("Aligned Diabetic categories:", df_merged['Diabetic'].unique())
    print("Aligned Race categories:", df_merged['Race'].unique())
    print("Aligned Smoking categories:", df_merged['Smoking'].unique())
    print("Aligned AgeCategory categories:", sorted(df_merged['AgeCategory'].unique()))
    print("Cleaning complete.\n")
    
    print("=== Step 4: Anomaly Handling & Outlier Mitigation ===")
    # Clip extreme BMI outliers to prevent model skewness (e.g., clip at 70)
    print(f"Original BMI max: {df_merged['BMI'].max()}")
    df_merged['BMI'] = df_merged['BMI'].clip(upper=70.0)
    print(f"Outlier-clipped BMI max: {df_merged['BMI'].max()}\n")
    
    print("=== Step 5: Smart Imputation (KNN Imputer) ===")
    numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    
    # Cast numeric columns properly
    for col in numeric_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        
    missing_count = df_merged[numeric_cols].isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values in numeric columns. Applying KNN Imputation...")
        # To prevent memory crash on 566k rows, we fit the KNN imputer on a representative sample of 10,000 rows
        sample_size = min(10000, len(df_merged))
        df_sample = df_merged[numeric_cols].dropna().sample(n=sample_size, random_state=42)
        
        imputer = KNNImputer(n_neighbors=5)
        imputer.fit(df_sample)
        
        # Transform the numeric columns
        df_merged[numeric_cols] = imputer.transform(df_merged[numeric_cols])
        print("KNN Imputation complete.")
    else:
        print("No missing values found in numeric columns. Skipping KNN Imputation.")
    print("")
    
    print("=== Step 6: Probabilistic Risk Scoring & Bias Mitigation ===")
    # Prepare features for the ML model
    cat_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DifficultyWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GeneralHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    # Create dummies for categoricals
    df_model = pd.get_dummies(df_merged, columns=cat_cols, drop_first=True)
    
    X = df_model.drop(columns=['HeartDisease', 'Year'])
    y = df_merged['HeartDisease'].map({'Yes': 1, 'No': 0})
    
    print(f"Training features shape: {X.shape}")
    
    # Train Logistic Regression model (natively outputs well-calibrated probabilities)
    print("Training Logistic Regression model...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X, y)
    
    # Mitigate 'Sick Quitter' bias:
    # Set the coefficient of AlcoholDrinking_Yes to 0.0 to neutralize its effect.
    # This prevents the model from predicting that alcohol consumption is a protective factor.
    adjusted = False
    for idx, col_name in enumerate(X.columns):
        if 'AlcoholDrinking_Yes' in col_name:
            original_coef = lr.coef_[0][idx]
            lr.coef_[0][idx] = 0.0
            print(f"Confounding Bias Mitigation: Adjusted '{col_name}' coefficient from {original_coef:.4f} to 0.0 (Neutral risk).")
            adjusted = True
            
    if not adjusted:
        print("Warning: Could not find 'AlcoholDrinking_Yes' column to perform bias adjustment.")
        
    # Generate predicted probabilities using the adjusted model
    print("Generating risk probabilities...")
    df_merged['HeartDisease_Probability'] = lr.predict_proba(X)[:, 1]
    
    # Standardize data types for final export
    df_merged['Year'] = df_merged['Year'].astype(int)
    print("Model predictions complete.\n")
    
    print("=== Step 7: Saving Processed Dataset ===")
    output_csv = 'processed_heart_data.csv'
    df_merged.to_csv(output_csv, index=False)
    print(f"Dataset successfully saved as '{output_csv}'\n")
    
    print("=== Step 8: Automated Data Verification ===")
    print(f"Saved file size: {os.path.getsize(output_csv) / (1024*1024):.2f} MB")
    
    # Verify shape
    print(f"Row count: {len(df_merged)} (Expected: 565817)")
    print(f"Column count: {len(df_merged.columns)} (Expected: 20)")
    
    # Verify column list
    expected_cols = [
        'HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
        'PhysicalHealth', 'MentalHealth', 'DifficultyWalking', 'Sex', 
        'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GeneralHealth', 
        'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Year', 
        'HeartDisease_Probability'
    ]
    missing_cols = [c for c in expected_cols if c not in df_merged.columns]
    if missing_cols:
        print("Warning: Missing columns in final output:", missing_cols)
    else:
        print("Verification: All expected columns are present.")
        
    # Check for NaNs
    nans = df_merged.isnull().sum().sum()
    print(f"NaN count: {nans} (Expected: 0)")
    
    # Check probability ranges
    prob_min = df_merged['HeartDisease_Probability'].min()
    prob_max = df_merged['HeartDisease_Probability'].max()
    print(f"Probability range: {prob_min:.4f} to {prob_max:.4f} (Expected: 0.0 to 1.0)")
    
    # Print sample outputs
    print("\nSample predicted risk scores:")
    print(df_merged[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'DifficultyWalking', 'GeneralHealth', 'HeartDisease_Probability']].head())

if __name__ == '__main__':
    main()
