import pandas as pd
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# Load the patient record dataset
data = pd.read_csv("./test/test5.csv")

# Identify known duplicates based on the rec_id column
duplicates = data[data["rec_id"].str.contains("-dup-")]

# Create a dictionary mapping each duplicate record to its corresponding original record
originals = {}
for i, row in duplicates.iterrows():
    original_id = row["rec_id"].replace("-dup-", "-org-")
    if original_id in originals:
        originals[original_id].append(i)
    else:
        originals[original_id] = [i]

# Create a new column called "match" to indicate whether a record is a duplicate or not
data["match"] = 0
for original_id, duplicates in originals.items():
    data.loc[duplicates, "match"] = 1

# Create separate label encoders for 'culture' and 'sex'
culture_encoder = LabelEncoder()
sex_encoder = LabelEncoder()

# Convert the categorical variables using label encoding
data["culture"] = culture_encoder.fit_transform(data["culture"])
data["sex"] = sex_encoder.fit_transform(data["sex"])

# Define the columns to be target encoded
name_cols = ["given_name", "surname", "street_number", "address_1", "address_2", "state", "phone_number", "national_identifier", "date_of_birth"]

# Define the random state
random_state = 42
#random_state = None

# Split the dataset into a training set and a testing set
train_data, test_data, train_labels, test_labels = train_test_split(
    data.drop(columns=["match", "rec_id"]), data["match"], test_size=0.9, random_state=random_state)
    #data.drop(columns=["match", "rec_id"]), data["match"], test_size=0.9, random_state=random_state)

# Instantiate a target encoder
encoder = ce.TargetEncoder(cols=name_cols)

# Fit the encoder on the training data and apply the transformation
train_data_encoded = encoder.fit_transform(train_data, train_labels)

# Apply the transformation to the test data
test_data_encoded = encoder.transform(test_data)

# Perform cross-validation using StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# Create the Random Forests model
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
# Hyperparameter tuning using GridSearchCV for the Random Forest model
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
rf_grid_search.fit(train_data_encoded, train_labels)
print("Best Random Forest parameters found:", rf_grid_search.best_params_)

# Create the Linear SVM model
svm = LinearSVC(random_state=random_state)
svm.fit(train_data_encoded, train_labels)

# Create the Gradient Boosted Trees model
gbt = GradientBoostingClassifier(random_state=random_state)
# Hyperparameter tuning using GridSearchCV for the Gradient Boosting model
gbt_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

gbt_grid_search = GridSearchCV(gbt, gbt_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
gbt_grid_search.fit(train_data_encoded, train_labels)
print("Best Gradient Boosting parameters found:", gbt_grid_search.best_params_)

# Train the models with the best hyperparameters
best_rf = rf_grid_search.best_estimator_
best_gbt = gbt_grid_search.best_estimator_

# Use the trained models to predict the match status of the patients in the testing set
rf_preds = best_rf.predict(test_data_encoded)
svm_preds = svm.predict(test_data_encoded)
gbt_preds = best_gbt.predict(test_data_encoded)

# Evaluate the accuracy of the models
rf_acc = accuracy_score(test_labels, rf_preds)
svm_acc = accuracy_score(test_labels, svm_preds)
gbt_acc = accuracy_score(test_labels, gbt_preds)

print("Random Forests accuracy:", rf_acc)
print("Linear SVM accuracy:", svm_acc)
print("Gradient Boosted Trees accuracy:", gbt_acc)

# Calculate cosine similarity between records in the testing set
similarity_matrix = cosine_similarity(test_data_encoded)

# Define the new record
# this record should match
new_record = {
    "rec_id": "rec-1006",
    "culture": "eng",
    "sex": "f",
    "given_name": "katherine",
    "surname": "sackvi lr",
    "street_number": 89,
    "address_1": "mcfarlan place",
    "state": "qld",
    "phone_number": "03218475",
    "national_identifier": "64036271",
    "blocking_number": 7,
    "date_of_birth": "19500708",
    "address_2": ""
}

# Convert the categorical variables in the new record using the same label encoder
new_record["culture"] = culture_encoder.transform([new_record["culture"]])[0]
new_record["sex"] = sex_encoder.transform([new_record["sex"]])[0]

# Convert the new record into a DataFrame
new_data = pd.DataFrame([new_record])

# Remove the 'rec_id' column from the new_data DataFrame
new_data = new_data.drop(columns=["rec_id"])

# Apply the target encoding transformation to the new record
new_data_encoded = encoder.transform(new_data)

# Make sure the columns in new_data_encoded are in the same order as in train_data_encoded
new_data_encoded = new_data_encoded[train_data_encoded.columns]

# Function to classify the new record based on the probability, the defined thresholds, and the model name
def classify_and_print_matched_record(prob, model_name):
    if prob >= match_threshold:
        print(f"Record match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")
        print(f"Matched with record:\n{train_data.iloc[most_similar_index]}\n")
    elif prob >= potential_match_threshold:
        print(f"Potential record match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")
        print(f"Matched with record:\n{train_data.iloc[most_similar_index]}\n")
    else:
        print(f"No match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")

# Set the match probability thresholds
match_threshold = 0.85
potential_match_threshold = 0.65

# Use the models to predict the match probabilities for the new record
rf_probs = best_rf.predict_proba(new_data_encoded)[:, 1]
svm_probs = svm.decision_function(new_data_encoded)
gbt_probs = best_gbt.predict_proba(new_data_encoded)[:, 1]

# Calculate the most similar index using cosine similarity
similarities = cosine_similarity(train_data_encoded, new_data_encoded)
most_similar_index = similarities.argmax()

# Classify the new record and print the matched record based on the probabilities and the defined thresholds for each model
print(f"\nRandom Forests Prob {rf_probs}:")
classify_and_print_matched_record(rf_probs[0], "Random Forests")
print(f"\nLinear SVM  {svm_probs}:")
classify_and_print_matched_record(svm_probs[0], "Linear SVM")
print(f"\nGradient Boosted Trees  {gbt_probs}:")
classify_and_print_matched_record(gbt_probs[0], "Gradient Boosted Trees")