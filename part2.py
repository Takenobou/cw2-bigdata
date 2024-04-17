import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import LeakyReLU
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline

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


def transform_readmitted_column(df):
    """Transform the readmitted column into binary (0, 1)."""
    df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)


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
            nn.Linear(input_size, 256),
            LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            LeakyReLU(0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_train_auc, valid_batches = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if len(np.unique(labels.cpu().numpy())) > 1:
                valid_batches += 1
                auc = roc_auc_score(labels.cpu().detach().numpy(), outputs.sigmoid().detach().cpu().numpy())
                total_train_auc += auc

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_auc = total_train_auc / valid_batches if valid_batches > 0 else 0
        train_losses.append(avg_train_loss)
        train_aucs.append(avg_train_auc)

        # Validation phase
        model.eval()
        total_val_loss, total_val_auc, valid_batches = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.float(), labels.float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                if len(np.unique(labels.cpu().numpy())) > 1:
                    valid_batches += 1
                    auc = roc_auc_score(labels.cpu().numpy(), outputs.sigmoid().cpu().numpy())
                    total_val_auc += auc

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_auc = total_val_auc / valid_batches if valid_batches > 0 else 0
        val_losses.append(avg_val_loss)
        val_aucs.append(avg_val_auc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train AUC: {avg_train_auc:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {avg_val_auc:.4f}')
        scheduler.step(avg_val_loss)

    return train_losses, val_losses, train_aucs, val_aucs


def preprocess_and_encode_data(df, numerical_features, categorical_features):
    """Preprocess and encode the data for model training and testing."""
    # Convert categorical features to string type to avoid type mismatch issues
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    # Separating the features and the target
    X = df[numerical_features + categorical_features]
    y = df['readmitted']

    # Define a pipeline for handling preprocessing steps
    preprocessing_pipeline = ImblearnPipeline([
        ('encoder', OneHotEncoder()),  # Encode categorical features
        ('scaler', StandardScaler(with_mean=False)),  # Scale numerical features, avoiding mean subtraction
        ('smote', SMOTE(random_state=42))  # Handle class imbalance
    ])

    # Process features and target
    X_processed, y_processed = preprocessing_pipeline.fit_resample(X, y)
    return X_processed, y_processed


def run_torch_with_smote_model(processed_data):
    transform_readmitted_column(processed_data)

    # Preprocess and encode data
    X_processed, y_processed = preprocess_and_encode_data(processed_data, numerical_features, categorical_features)
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y_processed, test_size=0.3, random_state=42,
                                                        stratify=y_processed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Convert sparse matrices to dense arrays before creating tensors
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_val_dense = X_val.toarray() if hasattr(X_val, 'toarray') else X_val
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

    # Convert pandas Series to numpy arrays explicitly and ensure correct data type
    y_train_array = y_train.to_numpy().astype(float)
    y_val_array = y_val.to_numpy().astype(float)
    y_test_array = y_test.to_numpy().astype(float)

    print("Training set class distribution:", np.bincount(y_train_array.astype(int)))
    print("Validation set class distribution:", np.bincount(y_val_array.astype(int)))

    # Create DataLoaders for both training and validation sets
    train_dataset = DiabetesDataset(torch.tensor(X_train_dense, dtype=torch.float32),
                                    torch.tensor(y_train_array, dtype=torch.float32))
    val_dataset = DiabetesDataset(torch.tensor(X_val_dense, dtype=torch.float32),
                                  torch.tensor(y_val_array, dtype=torch.float32))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Corrected model initialization
    model = BinaryClassificationModel(input_size=X_train_dense.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train and validate the model
    train_losses, val_losses, train_aucs, val_aucs = train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting AUC
    plt.figure(figsize=(10, 5))
    plt.plot(train_aucs, label='Training AUC')
    plt.plot(val_aucs, label='Validation AUC')
    plt.title('Training and Validation AUC per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    from sklearn.metrics import classification_report, confusion_matrix

    # Assuming `y_test` and `model` are available
    y_pred = model(torch.tensor(X_test_dense).float()).detach().numpy()
    y_pred = (y_pred > 0.5).astype(int)  # Assuming a binary classification with a threshold of 0.5

    print(classification_report(y_test_array, y_pred))
    conf_matrix = confusion_matrix(y_test_array, y_pred)

    # Plotting the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    file_path = 'diabetic_data.csv'
    sns.set(style="whitegrid")

    processed_data, shape_before, shape_after, numerical_features, categorical_features = preprocess_data(file_path)
    run_torch_with_smote_model(processed_data)

