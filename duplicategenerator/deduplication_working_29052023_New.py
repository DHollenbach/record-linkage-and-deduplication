import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import joblib
import re
from sklearn.metrics import f1_score

#nltk.download('punkt')

# Load the patient record dataset
data = pd.read_csv("./test/test5.csv")

# Combine 'street_number', 'address_1', and 'address_2' into one field and clean it
data['address'] = data['street_number'].astype(str) + ' ' + data['address_1'].astype(str) + ' ' + data['address_2'].astype(str)
data['address'] = data['address'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

# Drop the 'blocking_number' field
data = data.drop(columns=['blocking_number', 'street_number', 'address_1', 'address_2'])

# Identify known duplicates based on the rec_id column
duplicates = data[data["rec_id"].str.contains("-dup-")]

# Create a dictionary mapping each duplicate record to its corresponding original record
originals = {}
for i, row in duplicates.iterrows():
    original_id = row["rec_id"].replace("-dup-0", "-org")
    if original_id in originals:
        originals[original_id].append(i)
    else:
        originals[original_id] = [i]

# Create a new column called "match" to indicate whether a record is a duplicate or not
data["match"] = 0
for original_id, duplicates in originals.items():
    data.loc[duplicates, "match"] = 1
    # this ensures that both the duplicate and the match are in the same training set
    data.loc[data['rec_id'] == original_id, 'match'] = 1

# Fill missing values for 'culture', 'sex', and 'state' with a default value
default_values = {'culture': 'unknown', 'sex': 'unknown', 'state': 'unknown'}
data.fillna(value=default_values, inplace=True)

# Remove extra spaces from string columns
string_columns = ['given_name', 'surname', 'address']
for col in string_columns:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Label encoding for 'culture', 'sex', and 'state'
label_encoder_columns = ['culture', 'sex', 'state']
for col in label_encoder_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col].astype(str).str.lower())

# Tokenize the string columns and store them in a new DataFrame
tokenized_data = pd.DataFrame()
for col in string_columns:
    tokenized_data[col] = data[col].apply(word_tokenize)

# Generate the Word2Vec embeddings for the tokenized string columns
vector_size = 250
model_w2v = Word2Vec(tokenized_data.values.flatten(), vector_size=vector_size, min_count=1, workers=4)

# Save the trained model
model_w2v.save("word2vec_properties.model")

# Define the weights for the string columns
weights = {
    'given_name': 0.92, 
    'surname': 0.92, 
    'address': 0.92, 
    'phone_number': 0.92, 
    'national_identifier': 0.92,
    'sex': 0.92,
    'date_of_birth': 0.92,
    'culture': 0.92,
    'state': 0.92
}

# Apply the Word2Vec embeddings to the tokenized string columns
embedded_data = pd.DataFrame()
for col in string_columns:
    embeddings = np.vstack(tokenized_data[col].apply(lambda x: np.mean([model_w2v.wv[token] for token in x], axis=0)).values)
    temp_df = pd.DataFrame(embeddings, columns=[f"{col}_embed_{i}" for i in range(embeddings.shape[1])])
    temp_df *= weights[col]  # Apply the weights
    embedded_data = pd.concat([embedded_data, temp_df], axis=1)

# Replace the original string columns with the embedded columns
data = pd.concat([data.drop(columns=string_columns), embedded_data], axis=1)

# Impute missing values for numerical columns with the mean value
numerical_columns = ['date_of_birth', 'phone_number', 'national_identifier']
for col in numerical_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Standardize numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Split the data into features (X) and labels (y)
X = data.drop(columns=['rec_id', 'match'])
y = data['match']

# Initialize the classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = LinearSVC(random_state=42)
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Perform Stratified K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_accuracies = []
svm_accuracies = []
gbm_accuracies = []

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train and evaluate the classifiers
    for clf, acc_list in [(rf, rf_accuracies), (svm, svm_accuracies), (gbm, gbm_accuracies)]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc_list.append(accuracy)

# Calculate the average accuracy for each classifier
rf_avg_accuracy = np.mean(rf_accuracies)
svm_avg_accuracy = np.mean(svm_accuracies)
gbm_avg_accuracy = np.mean(gbm_accuracies)

print("Random Forest Average Accuracy: {:.2f}".format(rf_avg_accuracy))
print("Support Vector Machine Average Accuracy: {:.2f}".format(svm_avg_accuracy))
print("Gradient Boosting Machine Average Accuracy: {:.2f}".format(gbm_avg_accuracy))

# Select the best model
best_model = None
best_accuracy = 0

for model, accuracy in [(rf, rf_avg_accuracy), (svm, svm_avg_accuracy), (gbm, gbm_avg_accuracy)]:
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

print("Best Model: {}".format(type(best_model).__name__))
print("Best Model Average Accuracy: {:.2f}".format(best_accuracy))

# Save the best model
joblib.dump(rf, 'best_rf_model.pkl')
joblib.dump(svm, 'best_svm_model.pkl')
joblib.dump(gbm, 'best_gbm_model.pkl')

# Save the fitted SimpleImputer and StandardScaler instances
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')