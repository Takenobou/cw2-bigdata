import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn import feature_selection, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping

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
    # engineer_features(df)
    # reduce_dimensions(df, n_components=0.95)
    drop_all_missing_rows(df)

    # Map diagnosis codes to categories
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col + '_category'] = df[col].apply(map_icd_codes_to_categories)

    numerical_features, categorical_features = identify_features(df)
    df = remove_outliers(df, numerical_features)
    remove_duplicate_patients(df)
    normalize_features(df)
    # df = apply_pca(df, n_components=0.95)

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


def model_test(mdl, X,y):
    """Test the model using the testing set."""
    y_pred = mdl.predict(X) if hasattr(mdl, "predict") else mdl.predict_proba(X)[:, 1] >= 0.5
    accuracy = metrics.accuracy_score(y, y_pred)
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    cross_val_scores = cross_val_score(mdl, X, y, cv=5)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {confusion_matrix}")
    print(f"Cross Validation Score: {cross_val_scores}")
    for t in [0.2, 0.3, 0.5, 0.1, 0.01, 0.08]:
        crosstab = classify_for_threshold(mdl, X, y, t)
        print("Threshold {}:\n{}\n".format(t, crosstab))
    prob = np.array(mdl.predict_proba(X)[:, 1])
    y += 1
    fpr, sensitivity, _ = metrics.roc_curve(y, prob, pos_label=2)
    print("AUC = {}".format(metrics.auc(fpr, sensitivity)))
    plt.scatter(fpr, fpr, c='b', marker='s')
    plt.scatter(fpr, sensitivity, c='r', marker='o')
    plt.show()


def classify_for_threshold(mdl, X, Y, t):
    prob_df = pd.DataFrame(mdl.predict_proba(X)[:, 1])
    prob_df['predict'] = np.where(prob_df[0] >= t, 1, 0)
    prob_df['actual'] = Y
    return pd.crosstab(prob_df['actual'], prob_df['predict'])


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
    r_features = r_features.copy()  # Ensure r_features is a copy, not a view
    r_features['readmitted'] = df.loc[:, df.columns == 'readmitted']
    return r_features


def find_optimal_clusters(data):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def engineer_features(df):
    """Further engineer features with advanced techniques."""
    # Base features
    df['medical_history_complexity'] = df[
        ['num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)

    scaler = StandardScaler()
    df['medical_history_complexity_standardized'] = scaler.fit_transform(df[['medical_history_complexity']])

    # Interaction features refinement
    df['medication_emergency_interaction'] = df['num_medications'] * np.log1p(df['number_emergency'])
    df['outpatient_inpatient_ratio'] = df['number_outpatient'] / (df['number_inpatient'] + 1)

    # Polynomial features
    pf = PolynomialFeatures(degree=2, include_bias=False)
    key_features = df[['num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']].copy()
    poly_features = pf.fit_transform(key_features)
    for i, col_name in enumerate(pf.get_feature_names_out(key_features.columns)):
        df[f'poly_{col_name}'] = poly_features[:, i]

    # Clustering-based features
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['medical_history_complexity_standardized', 'num_medications']])

    # Dynamic quantile for high risk with refined approach
    for threshold in [0.6, 0.7, 0.8]:
        df[f'high_risk_{threshold}'] = np.where(
            df['medical_history_complexity_standardized'] > df['medical_history_complexity_standardized'].quantile(
                threshold), 1, 0)

    # Categorical risk level
    df['risk_level'] = pd.qcut(df['medical_history_complexity_standardized'], q=[0, 0.25, 0.5, 0.75, 1],
                               labels=['low', 'medium', 'high', 'very_high'])

    return df


def reduce_dimensions(df, n_components=0.95):
    # Apply PCA to reduce dimensions
    features = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    df_pca = pd.DataFrame(data=principal_components)
    df.reset_index(drop=True, inplace=True)
    df_pca.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_pca], axis=1)
    return df


def normalize_features(df, exclude_columns=['age']):
    """Normalize numerical features."""

    numerical_features = df.select_dtypes(include=[np.number]).columns
    features_to_normalize = numerical_features.difference(exclude_columns)
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])


def apply_pca(df, n_components=None):
    # Separate out the numerical data
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_data = df[numerical_features]

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    principalComponents = pca.fit_transform(numerical_data)

    # Convert to dataframe and concatenate with original df
    pca_df = pd.DataFrame(data=principalComponents, columns=[f'pc_{i}' for i in range(principalComponents.shape[1])])
    df.reset_index(drop=True, inplace=True)  # Reset index to avoid concatenation issues
    df = pd.concat([df, pca_df], axis=1)

    return df


def normalize_data_min_max(df, numerical_features):
    """Normalize numerical features using Min-Max Scaling."""
    df_numerical = df[numerical_features]
    df_normalized = (df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())
    df[df_numerical.columns] = df_normalized
    return df


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
    processed_data = normalize_data_min_max(processed_data, numerical_features)

    # Data exploration
    transform_readmitted_column(processed_data)
    plot_age_impact(processed_data)
    plot_race_impact(processed_data)
    plot_gender_impact(processed_data)
    plot_diagnosis_category_impact(processed_data)

    # Prepare the dataset for model building
    subset = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient',
              'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'readmitted']
    rfe_subset_data = recursive_feature_elimination(processed_data[subset])
    rfe_test_set, rfe_training_set = split_data(rfe_subset_data)
    test_set, training_set = split_data(processed_data[subset])

    # Handling imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(training_set.drop('readmitted', axis=1), training_set['readmitted'])
    X_test, y_test = test_set.drop('readmitted', axis=1), test_set['readmitted']
    X_rfe_train, y_rfe_train = sm.fit_resample(rfe_training_set.drop('readmitted', axis=1), rfe_training_set['readmitted'])
    X_rfe_test, y_rfe_test = rfe_test_set.drop('readmitted', axis=1), rfe_test_set['readmitted']

    # Train and test the Logistic Regression Model
    print("-" * 20)
    print("Training and testing Logistic Regression Model...")
    lr_model = LogisticRegression()
    lr_model = model_train(lr_model, X_rfe_train, y_rfe_train)
    print("Model training completed.")

    # Test Logistic Regression Model
    model_test(lr_model, X_rfe_test, y_rfe_test)
    print("-" * 20)

    # Train and test the Random Forest Model
    print("Training and testing Random Forest Model...")
    crf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=30, random_state=42, oob_score=True)
    crf.fit(X_train, y_train)
    print("Model training completed.")
    print("Accuracy score for training data is: {:4.3f}".format(crf.score(X_train, y_train)))
    print("Accuracy score for test data: {:4.3f}".format(crf.score(X_train, y_train)))
    print("The Oob score is: {:4.3f}".format(crf.oob_score_))

    # Test Random Forest Model
    y_pred_rf = crf.predict(X_test)
    # print("Accuracy of Random Forest Model:", accuracy_score(y_test, y_pred_rf))

    ######################################################################
    test_set, training_set = split_data(df)

    test_X = test_set.iloc[:, :-1]
    test_y = test_set['readmitted']

    X = training_set.iloc[:, :-1]
    y = training_set['readmitted']
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled,
                                                      train_size=0.5,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=True)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    test_X = scaler.transform(test_X)



    model = keras.Sequential(
        [
            keras.layers.Dense(units=4, activation="relu", input_shape=(X_train.shape[-1],)),
            # randomly delete 30% of the input units below
            keras.layers.Dropout(0.4),
            keras.layers.Dense(units=4, activation="relu"),
            # the output layer, with a single neuron
            keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    initial_weights = model.get_weights()
    model.summary()

    learning_rate = 0.001
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=keras.metrics.AUC()
                  )

    early_stopping = EarlyStopping(
        min_delta=0.0002,
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=2000,
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[early_stopping])

    logs = pd.DataFrame(history.history)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(logs.loc[5:, "loss"], lw=2, label='training loss')
    plt.plot(logs.loc[5:, "val_loss"], lw=2, label='validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(logs.loc[5:, "auc"], lw=2, label='training ROC AUC score')
    plt.plot(logs.loc[5:, "val_auc"], lw=2, label='validation ROC AUC score')
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend(loc='lower right')
    plt.show()

    results = model.evaluate(test_X, test_y, batch_size=1000)
    print("test loss, test acc:", results)

    # y_predictions = model.predict(test_X)
    # accuracy = metrics.accuracy_score(y, y_predictions)
    # confusion_matrix = metrics.confusion_matrix(y, y_predictions)
    # print(f"Accuracy: {accuracy}")
    # print(f"Confusion Matrix:\n {confusion_matrix}")
