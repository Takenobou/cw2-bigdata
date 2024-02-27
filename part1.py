import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def replace_missing_values(df):
    """Replace custom missing values with np.nan."""
    df.replace('?', np.nan, inplace=True)


def drop_high_missing_columns(df, threshold=0.5):
    """Drop columns with missing values above a specified threshold."""
    thresh = len(df) * threshold
    df.dropna(axis=1, thresh=thresh, inplace=True)


def drop_low_variance_columns(df, threshold=0.95):
    """Drop columns where a large percentage of the values are the same."""
    for col in df.columns:
        if df[col].value_counts(normalize=True).iloc[0] > threshold:
            df.drop(col, axis=1, inplace=True)


def transform_age_ranges(df):
    """Transform age ranges to the middle value of each range."""
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)


def replace_missing_diagnoses(df):
    """Replace missing values in diagnosis columns with 0."""
    df[['diag_1', 'diag_2', 'diag_3']] = df[['diag_1', 'diag_2', 'diag_3']].fillna(0)


def drop_all_missing_rows(df):
    """Drop all rows with any missing values."""
    df.dropna(inplace=True)


def identify_features(df):
    """Identify numerical and categorical features."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical_features, categorical_features


def remove_outliers(df, numerical_features):
    """Remove outliers from numerical columns."""
    for feature in numerical_features:
        mean = df[feature].mean()
        std = df[feature].std()
        df = df[np.abs(df[feature] - mean) <= (3 * std)]
    return df


def remove_duplicate_patients(df):
    """Remove duplicate patient entries."""
    df.drop_duplicates(subset=['patient_nbr'], inplace=True)


def map_icd_codes_to_categories(diag):
    """Map ICD diagnosis codes to primary diagnosis categories."""
    if pd.isnull(diag):
        return 'Other'
    try:
        code = float(diag)
        if 390 <= code <= 459 or code == 785:
            return 'A disease of the circulatory system'
        elif 250 <= code < 251:
            return 'Diabetes'
        elif 460 <= code <= 519 or code == 786:
            return 'A disease of the respiratory system'
        elif 520 <= code <= 579 or code == 787:
            return 'Diseases of the digestive system'
        elif 800 <= code <= 999:
            return 'Injury and poisoning'
        elif 710 <= code <= 739:
            return 'Diseases of the musculoskeletal system and connective tissue'
        elif 580 <= code <= 629 or code == 788:
            return 'Diseases of the genitourinary system'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'


def preprocess_data(file_path):
    df = load_data(file_path)
    shape_before = df.shape

    replace_missing_values(df)
    drop_high_missing_columns(df)
    drop_low_variance_columns(df)
    transform_age_ranges(df)
    replace_missing_diagnoses(df)
    drop_all_missing_rows(df)

    # Map diagnosis codes to categories
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col + '_category'] = df[col].apply(map_icd_codes_to_categories)

    numerical_features, categorical_features = identify_features(df)
    df = remove_outliers(df, numerical_features)
    remove_duplicate_patients(df)

    shape_after = df.shape
    return df, shape_before, shape_after, numerical_features, categorical_features


def plot_age_impact(df):
    """Plot the impact of age on readmission."""
    plt.figure(figsize=(12, 6))
    age_plot = sns.barplot(x='age', y='readmitted', data=df, palette="coolwarm", hue='age', legend=False)
    age_plot.set_title('Impact of Age on Readmission Rates')
    age_plot.set_xlabel('Age Group')
    age_plot.set_ylabel('Readmission Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_race_impact(df):
    """Plot the impact of race on readmission."""
    plt.figure(figsize=(12, 6))
    race_plot = sns.barplot(x='race', y='readmitted', data=df, palette="muted", hue='race', legend=False)
    race_plot.set_title('Readmission Rates by Race')
    race_plot.set_xlabel('Race')
    race_plot.set_ylabel('Readmission Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_gender_impact(df):
    """Plot the impact of gender on readmission."""
    plt.figure(figsize=(12, 6))
    gender_plot = sns.barplot(x='gender', y='readmitted', data=df, palette="pastel", hue='gender', legend=False)
    gender_plot.set_title('Readmission Rates by Gender')
    gender_plot.set_xlabel('Gender')
    gender_plot.set_ylabel('Readmission Rate')
    plt.tight_layout()
    plt.show()


def transform_readmitted_column(df):
    """Transform the readmitted column into binary (0, 1)."""
    df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)


def plot_diagnosis_impact(df, diagnosis_column='diag_1'):
    """Plot the impact of diagnosis types on readmission."""
    plt.figure(figsize=(14, 7))
    diag_plot = sns.countplot(x=diagnosis_column, hue='readmitted', data=df, palette="deep")
    diag_plot.set_title('Readmission Rates by Diagnosis Type')
    diag_plot.set_xlabel('Diagnosis Type')
    diag_plot.set_ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_diagnosis_category_impact(df):
    """Plot the impact of diagnosis categories on readmission."""
    plt.figure(figsize=(14, 7))
    diag_plot = sns.countplot(x='diag_1_category', hue='readmitted', data=df, palette="deep")
    diag_plot.set_title('Readmission Rates by Primary Diagnosis Category')
    diag_plot.set_xlabel('Primary Diagnosis Category')
    diag_plot.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = 'diabetic_data.csv'
    sns.set(style="whitegrid")

    # Data cleaning and transformation
    processed_data, shape_before, shape_after, numerical_features, categorical_features = preprocess_data(file_path)
    print(f"Shape before preprocessing: {shape_before}")
    print(f"Shape after preprocessing: {shape_after}")
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Data exploration
    transform_readmitted_column(processed_data)
    plot_age_impact(processed_data)
    plot_race_impact(processed_data)
    plot_gender_impact(processed_data)
    plot_diagnosis_category_impact(processed_data)
