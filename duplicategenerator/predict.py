import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from nltk.tokenize import word_tokenize
import numpy as np
import joblib
from gensim.models import Word2Vec
import re

string_columns = ['given_name', 'surname', 'address']
label_encoder_columns = ['culture', 'sex', 'state']
numerical_columns = ['date_of_birth', 'phone_number', 'national_identifier']
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
trained_model = Word2Vec.load("word2vec_properties.model")

# Load the fitted SimpleImputer and StandardScaler instances
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Define the weights for the string columns
# typos match
weights = {
    'given_name': .92, 
    'surname': .92, 
    'address': .92, 
    'phone_number': .92, 
    'national_identifier': .92,
    'sex': .92,
    'date_of_birth': .92,
    'culture': .92,
    'state': .92
}

new_record = {
    "rec_id": "rec-0-org",
    "culture": "eng",
    "sex": "f",
    "given_name": "elin",
    "surname": "snedlry",
    "street_number": 62,
    "address_1": "bunurung close",
    "address_2": "",
    "state": "nsw",
    "date_of_birth": "1926022",
    "phone_number": "61271389",
    "national_identifier": "66254184"
}

# Exact Match
# weights = {
#     'given_name': 0.6, 
#     'surname': 1, 
#     'address': 1, 
#     'phone_number': 1, 
#     'national_identifier': 1,
#     'sex': 1,
#     'date_of_birth': 1,
#     'culture': 1,
#     'state': 1
# }

# new_record = {
#     "rec_id": "rec-0-org",
#     "culture": "eng",
#     "sex": "f",
#     "given_name": "smedley",
#     "surname": "snedlry",
#     "street_number": 62,
#     "address_1": "bunurung close",
#     "address_2": "",
#     "state": "nsw",
#     "date_of_birth": "19260222",
#     "phone_number": "612713892",
#     "national_identifier": "66254184"
# }

# preprocess the new record
new_data = pd.DataFrame([new_record])

# Combine 'street_number', 'address_1', and 'address_2' into one field and clean it
new_data['address'] = new_data['street_number'].astype(str) + ' ' + new_data['address_1'].astype(str) + ' ' + new_data['address_2'].astype(str)
new_data['address'] = new_data['address'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

# Drop the 'street_number', 'address_1', 'address_2' fields
new_data = new_data.drop(columns=['street_number', 'address_1', 'address_2'])

# Remove the 'rec_id' column from the new_data DataFrame
new_data = new_data.drop(columns=["rec_id"])

# Clean the data
# Fill missing values for 'culture', 'sex', and 'state' with a default value
default_values = {'culture': 'unknown', 'sex': 'unknown', 'state': 'unknown'}
new_data.fillna(value=default_values, inplace=True)

# Remove extra spaces from string columns
for col in string_columns:
    new_data[col] = new_data[col].astype(str).str.strip()

# Label encoding for 'culture', 'sex', and 'state'
for col in label_encoder_columns:
    label_encoder = LabelEncoder()
    new_data[col] = label_encoder.fit_transform(new_data[col].astype(str))

# Tokenize the string columns and store them in a new DataFrame
tokenized_new_data = pd.DataFrame()
for col in string_columns:
    tokenized_new_data[col] = new_data[col].apply(word_tokenize)

# Update the vocabulary with new data
trained_model.build_vocab(tokenized_new_data, update=True)

# Train the model on the new data
trained_model.train(tokenized_new_data, total_examples=len(tokenized_new_data), epochs=trained_model.epochs)

# Apply the Word2Vec embeddings to the tokenized string columns
embedded_new_data = pd.DataFrame()
for col in string_columns:
    embeddings = np.vstack(tokenized_new_data[col].apply(lambda x: np.mean([trained_model.wv[token] if token in trained_model.wv else np.zeros(trained_model.vector_size) for token in x], axis=0)).values)
    temp_df = pd.DataFrame(embeddings, columns=[f"{col}_embed_{i}" for i in range(embeddings.shape[1])])
    temp_df *= weights[col]  # Apply the weights
    embedded_new_data = pd.concat([embedded_new_data, temp_df], axis=1)

# Replace the original string columns with the embedded columns
new_data = pd.concat([new_data.drop(columns=string_columns), embedded_new_data], axis=1)

# Convert 'date_of_birth' to a numeric format and fill invalid values with NaN
for col in numerical_columns:
    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

# Impute missing values for numerical columns with the mean value
new_data[numerical_columns] = imputer.transform(new_data[numerical_columns])

# Standardize numerical columns
new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

print(new_data)

# Load the best model
best_rf_model = joblib.load('best_rf_model.pkl')
best_svm_model = joblib.load('best_svm_model.pkl')
best_gbm_model = joblib.load('best_gbm_model.pkl')

# Predict the outcome for the new record using the best model
rf_prediction = best_rf_model.predict(new_data)
svm_prediction = best_svm_model.predict(new_data)
gbm_prediction = best_gbm_model.predict(new_data)

# Get the predicted class probabilities
rf_proba = best_rf_model.predict_proba(new_data)[:, 1]
svm_proba = best_svm_model.decision_function(new_data)
gbm_proba = best_gbm_model.predict_proba(new_data)[:, 1]

print("Random Forest Prediction: {}, Prob {}".format(rf_prediction[0], rf_proba))
print("Support Vector Machine Prediction: {}, Prob {}".format(svm_prediction[0], svm_proba))
print("Gradient Boosting Machine Prediction: {}, Prob {}".format(gbm_prediction[0], gbm_proba))

trained_model.save("word2vec_properties_updated.model")