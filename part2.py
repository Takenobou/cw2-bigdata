import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR


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


def transform_data_for_model(df, numerical_features, target_column='readmitted'):
    """Prepare data for model training and testing."""
    X = df[numerical_features].values
    y = df[target_column].values
    return X, y


class DiabetesDataset(torch.utils.data.Dataset):
    """Diabetes dataset."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss, total_train_auc = 0, 0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            auc = roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            total_train_auc += auc

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_auc = total_train_auc / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss, total_val_auc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.float(), labels.float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                auc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
                total_val_auc += auc

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_auc = total_val_auc / len(val_loader)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Train AUC: {avg_train_auc:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}, '
            f'Val AUC: {avg_val_auc:.4f}')

        scheduler.step()


if __name__ == "__main__":
    file_path = 'diabetic_data.csv'
    sns.set(style="whitegrid")

    processed_data, shape_before, shape_after, numerical_features, categorical_features = preprocess_data(file_path)
    transform_readmitted_column(processed_data)
    X, y = transform_data_for_model(processed_data, numerical_features)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create DataLoaders for both training and validation sets
    train_dataset = DiabetesDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.float32))
    val_dataset = DiabetesDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                  torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64,
                            shuffle=False)

    # Define model, criterion, and optimiser with a smaller learning rate
    model = BinaryClassificationModel(input_size=X_train_scaled.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Train and validate the model
    train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=80)
