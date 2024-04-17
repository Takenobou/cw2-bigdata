import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from distributed.protocol import torch
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, preprocessing
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn import feature_selection, metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder, OneHotEncoder, \
    RobustScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from xgboost import XGBClassifier


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
    # Make sure df['age'] contains the correct age group values, not normalized ones
    age_plot = sns.barplot(x='age', y='readmitted', data=df, palette="coolwarm")
    age_plot.set_title('Impact of Age on Readmission Rates')
    age_plot.set_xlabel('Age Group')
    age_plot.set_ylabel('Readmission Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_race_impact(df):
    """Plot the impact of race on readmission."""
    plt.figure(figsize=(12, 6))
    race_plot = sns.barplot(x='race', y='readmitted', data=df, palette="muted", hue='race')
    race_plot.set_title('Readmission Rates by Race')
    race_plot.set_xlabel('Race')
    race_plot.set_ylabel('Readmission Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_gender_impact(df):
    """Plot the impact of gender on readmission."""
    plt.figure(figsize=(12, 6))
    gender_plot = sns.barplot(x='gender', y='readmitted', data=df, palette="pastel", hue='gender')
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


def split_data(pd):
    """Split the data into training and testing sets."""
    # test = processed_data.sample(frac=0.2, random_state=42)
    # train = processed_data.drop(test.index)

    train, test = train_test_split(pd, test_size=0.2, shuffle=True, random_state=93, stratify=pd['readmitted'])
    print(f"Train shape: {train.shape}")
    print(f"Readmitted in Train set:\n {train['readmitted'].value_counts()}")
    print(f"Percentage of Readmitted in Train set:\n {train['readmitted'].value_counts(normalize=True)}")
    print(f"Test shape: {test.shape}")
    print(f"Readmitted in Test set:\n {test['readmitted'].value_counts()}")
    print(f"Percentage of Readmitted in Test set:\n {test['readmitted'].value_counts(normalize=True)}")
    return test, train


def model_train(mdl, X, y):
    """Train the model using the training set."""
    mdl.fit(X, y)

    # Check if the model is a type of linear model
    if hasattr(mdl, 'intercept_'):
        print(f'Intercept: {mdl.intercept_}\n')
    if hasattr(mdl, 'coef_'):
        print(f'Coefficients: {mdl.coef_}')
        for feat, coef in zip(X.columns, mdl.coef_[0]):
            print(f"{feat}: {coef}")
        print(f"Model score against training data: {mdl.score(X, y)}")

    # For Random Forest or other tree-based models, you might want to print feature importances instead
    if hasattr(mdl, 'feature_importances_'):
        print("Feature importances:", mdl.feature_importances_)

    return mdl


def model_test(mdl, X, y):
    """Test the model using the testing set."""
    y_pred = mdl.predict_proba(X)[:, 1] >= 0.56
    accuracy = metrics.accuracy_score(y, y_pred)
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {confusion_matrix}")
    return mdl

def plot_roc_curve(mdl, X, y):
    prob = np.array(mdl.predict_proba(X)[:, 1])
    y += 1
    fpr, sensitivity, _ = metrics.roc_curve(y, prob, pos_label=2)
    print("AUC = {}".format(metrics.auc(fpr, sensitivity)))
    plt.scatter(fpr, fpr, c='b', marker='s')
    plt.scatter(fpr, sensitivity, c='r', marker='o')
    plt.show()


def cross_validate_model(mdl, X, y):
    """Cross-validate the model using the provided set."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(mdl, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated scores: {scores}")
    print(f"Mean accuracy: {scores.mean()}")


def recursive_feature_elimination(df):
    """Apply recursive feature elimination to the dataset."""
    X = df.loc[:, df.columns != 'readmitted']
    y = df['readmitted']
    # Standardize the features
    standardizer = StandardScaler()
    X0 = standardizer.fit_transform(X)
    X0 = pd.DataFrame(X0, index=X.index, columns=X.columns)
    # Apply RFE
    mod = linear_model.LogisticRegression(max_iter=1000)
    selector = feature_selection.RFE(mod, n_features_to_select=6, verbose=1, step=1)
    selector = selector.fit(X0, y)
    r_features = X.loc[:, selector.support_]
    print("R features are:\n{}".format(','.join(list(r_features))))
    r_features = r_features.copy()
    r_features['readmitted'] = df.loc[:, df.columns == 'readmitted']
    return r_features


def elbow_method(df, numerical_features):
    """Use the elbow method to determine the optimal number of clusters."""
    data = df[numerical_features]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, max_iter=300, n_init='auto')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()




def apply_kmeans(df, num_clusters):
    # Scaling the features before applying K-Means
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_features])  # Assuming 'numerical_features' is defined

    # Apply K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(df_scaled)
    df['Cluster'] = clusters

    # Analyze the cluster centroids
    centroids = kmeans.cluster_centers_
    print("Centroids of clusters:")
    print(centroids)

    # Plotting the distribution of clusters
    sns.countplot(x='Cluster', data=df)
    plt.title('Distribution of Clusters')
    plt.show()

    return df, centroids


def normalize_data_min_max(df, numerical_features):
    """Normalize numerical features using Min-Max Scaling."""
    df_numerical = df[numerical_features]
    df_normalized = (df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())
    df[df_numerical.columns] = df_normalized
    return df


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
            nn.Dropout(0.2),
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

    # Data cleaning and transformation
    processed_data, shape_before, shape_after, numerical_features, categorical_features = preprocess_data(file_path)

    print(f"Shape before preprocessing: {shape_before}")
    print(f"Shape after preprocessing: {shape_after}")
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Normalize data
    # processed_data = normalize_data_min_max(processed_data, numerical_features)

    # Data exploration
    transform_readmitted_column(processed_data)
    # plot_age_impact(processed_data)
    # plot_race_impact(processed_data)
    # plot_gender_impact(processed_data)
    # plot_diagnosis_category_impact(processed_data)
    # elbow_method(processed_data, numerical_features)


    # Prepare the dataset for model building
    subset = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient',
              'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'readmitted']
    rfe_subset_data = recursive_feature_elimination(processed_data[subset])
    rfe_test_set, rfe_training_set = split_data(rfe_subset_data)

    df = pd.get_dummies(processed_data)
    features_to_drop = df.nunique()
    features_to_drop = features_to_drop.loc[features_to_drop.values == 1].index
    df = df.drop(features_to_drop, axis=1)

    test_set, training_set = split_data(pd.get_dummies(processed_data))

    # # Handling imbalance with SMOTE
    # sm = SMOTE(random_state=96)
    # X_train, y_train = sm.fit_resample(training_set.drop('readmitted', axis=1), training_set['readmitted'])
    # X_test, y_test = test_set.drop('readmitted', axis=1), test_set['readmitted']
    # X_rfe_train, y_rfe_train = sm.fit_resample(rfe_training_set.drop('readmitted', axis=1),
    #                                            rfe_training_set['readmitted'])
    # X_rfe_test, y_rfe_test = rfe_test_set.drop('readmitted', axis=1), rfe_test_set['readmitted']
    # X_rfe_set, y_rfe_set = rfe_subset_data.drop('readmitted', axis=1), rfe_subset_data['readmitted']
    #
    # # Train and test the Logistic Regression Model
    # print("-" * 20)
    # print("Training and testing Logistic Regression Model...")
    # lr_model = LogisticRegression()
    # lr_model = model_train(lr_model, X_rfe_train, y_rfe_train)
    # print("Model training completed.")
    #
    # # Test Logistic Regression Model
    # lr_model=model_test(lr_model, X_rfe_test, y_rfe_test)
    # cross_validate_model(lr_model, X_rfe_set, y_rfe_set)
    # plot_roc_curve(lr_model, X_rfe_test, y_rfe_test)


    print("-" * 40)
    df_crs = processed_data
    shape_of_data = df_crs.shape
    print(shape_of_data)

    # Before you start encoding categorical features, convert them to strings
    for col in df_crs.select_dtypes(include=['object']).columns:
        df_crs[col] = df_crs[col].astype(str)

    # Apply Polynomial Features to Numerical Features
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_transformer.fit_transform(df_crs[numerical_features])
    poly_feature_names = poly_transformer.get_feature_names_out(numerical_features)

    # Drop the original numerical columns and add the polynomial features
    df_crs.drop(columns=numerical_features, inplace=True)
    df_crs_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=df_crs.index)
    df_crs = pd.concat([df_crs, df_crs_poly], axis=1)

    # Encode categorical features
    le = LabelEncoder()
    for col in df_crs.select_dtypes(include=['object']).columns:
        df_crs[col] = le.fit_transform(df_crs[col])

    # Separate features and target
    X_crf = df_crs.drop('readmitted', axis=1)
    y_crf = df_crs['readmitted']

    # Apply SMOTE to address class imbalance
    sm = SMOTEENN(random_state=42)
    X_res_crf, y_res_crf = sm.fit_resample(X_crf, y_crf)

    X_train_crf, X_test_crf, y_train_crf, y_test_crf = train_test_split(X_res_crf, y_res_crf, test_size=0.25,
                                                                        stratify=y_res_crf, random_state=69)

    # Proceed with training and evaluation
    crf = RandomForestClassifier(n_jobs=-1, n_estimators=400, min_samples_leaf=5, oob_score=True,
                                 criterion='log_loss', max_depth=10, random_state=42)
    crf.fit(X_train_crf, y_train_crf)

    # Evaluation
    train_score = crf.score(X_train_crf, y_train_crf)
    test_score = crf.score(X_test_crf, y_test_crf)

    print("Training set shape:", X_train_crf.shape, y_train_crf.shape)
    print("Test set shape:", X_test_crf.shape, y_test_crf.shape)

    print(f"Train score: {train_score:4.3f}")
    print(f"Test score: {test_score:4.3f}")
    print(f"The Oob score of the trained model is: {crf.oob_score_:4.3f}")

    # # Normalize data
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(df_crs.drop('readmitted', axis=1))
    #
    # # Applying PCA to reduce dimensions for visualization
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_scaled)
    #
    # # Applying Kmeans
    # kmeans = KMeans(n_clusters=3, random_state=42)
    # y_kmeans = kmeans.fit_predict(X_pca)
    #
    # # Visualize clusters
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    # plt.title('Clusters visualization in 2D PCA-reduced space')
    # plt.show()
    #
    # # Compare cluster distribution with the readmission feature
    # df_crs['cluster'] = y_kmeans
    # cluster_comparison = pd.crosstab(df_crs['cluster'], df_crs['readmitted'])
    # print(cluster_comparison)
    #
    # age_cluster_cross_tab = pd.crosstab(index=[df_crs['cluster'], df_crs['readmitted']], columns=df_crs['age'])
    # print(age_cluster_cross_tab)
    #
    # age_cluster_cross_tab.plot(kind='bar', stacked=True, figsize=(10, 7))
    # plt.title('Age Distribution by Cluster and Readmission')
    # plt.xlabel('Cluster and Readmission')
    # plt.ylabel('Count')
    # plt.show()
    #
    # race_mapping = {
    #     0: 'Caucasian',
    #     1: 'AfricanAmerican',
    #     2: 'Hispanic',
    #     3: 'Asian',
    #     4: 'Other'}
    #
    # race_cluster_cross_tab = pd.crosstab(index=[df_crs['cluster'], df_crs['readmitted']], columns=df_crs['race'])
    # race_cluster_cross_tab.rename(columns=race_mapping, inplace=True)
    # print(race_cluster_cross_tab)
    #
    # race_cluster_cross_tab.plot(kind='bar', stacked=True, figsize=(10, 7))
    # plt.title('Race Distribution by Cluster and Readmission')
    # plt.xlabel('Cluster and Readmission')
    # plt.ylabel('Count')
    # plt.legend(title='Race')
    # plt.show()

    # Apply cross-validation on the resampled data
    crossvalidation_crf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(crf, X_res_crf, y_res_crf, scoring='accuracy', cv=crossvalidation_crf)

    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score for Random Forest: {:.4f}".format(np.mean(cv_scores)))

    # Prediction on the test set
    y_test_pred = crf.predict(X_test_crf)
    y_train_pred = crf.predict(X_train_crf)

    # Classification report and confusion matrix
    print("Classification report for test:\n", classification_report(y_test_crf, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_crf, y_test_pred))

    # ROC-AUC score
    roc_auc = roc_auc_score(y_test_crf, crf.predict_proba(X_test_crf)[:, 1])
    print("ROC-AUC Score:", roc_auc)

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_crf, crf.predict_proba(X_test_crf)[:, 1])

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # # BCE model
    # X, y = transform_data_for_model(processed_data, numerical_features)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    #
    # # Normalise features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    # X_test_scaled = scaler.transform(X_test)
    #
    # # Create DataLoaders for both training and validation sets
    # train_dataset = DiabetesDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
    #                                 torch.tensor(y_train, dtype=torch.float32))
    # val_dataset = DiabetesDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
    #                               torch.tensor(y_val, dtype=torch.float32))
    #
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=64,
    #                         shuffle=False)
    #
    # # Define model, criterion, and optimiser with a smaller learning rate
    # model = BinaryClassificationModel(input_size=X_train_scaled.shape[1])
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    #
    # # Train and validate the model
    # train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30)


