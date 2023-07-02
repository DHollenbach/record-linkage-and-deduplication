import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Download the NLTK tokenizer only when needed
# Uncomment the following line to download the tokenizer
#nltk.download('punkt')

# Load the patient record dataset
data = pd.read_csv("./test/test5.csv")

# Identify known duplicates based on the rec_id column
duplicates = data[data["rec_id"].str.contains("-dup-")]

# Create a dictionary mapping each duplicate record to its corresponding original record
originals = {}
for i, row in duplicates.iterrows():
    # issue was that -dup-0 replacement was incorrect
    # ensure the original id is correct and in the format of
    # duplicate = rec-93-dup-0 & original = rec-93-org
    original_id = row["rec_id"].replace("-dup-0", "-org")
    if original_id in originals:
        originals[original_id].append(i)
    else:
        originals[original_id] = [i]

# Create a new column called "match" to indicate whether a record is a duplicate or not
data["match"] = 0
for original_id, duplicates in originals.items():
    data.loc[duplicates, "match"] = 1

# Fill missing values for 'culture', 'sex', and 'state' with a default value
default_values = {'culture': 'unknown', 'sex': 'unknown', 'state': 'unknown'}
data.fillna(value=default_values, inplace=True)

# Remove extra spaces from string columns
string_columns = ['given_name', 'surname', 'address_1', 'address_2', 'phone_number', 'national_identifier']
for col in string_columns:
    data[col] = data[col].astype(str).str.strip()

# Label encoding for 'culture', 'sex', and 'state'
label_encoder_columns = ['culture', 'sex', 'state']
for col in label_encoder_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Feature hashing for string columns
hasher = FeatureHasher(n_features=10, input_type='string')
hashed_columns = ['given_name', 'surname', 'address_1', 'address_2', 'phone_number', 'national_identifier']

# Use Word2Vec to encode string columns
text = ''

for col in hashed_columns:
    text += '_'.join(data[col].astype(str).fillna('')) + '_'

tokenized_text = [sentence.split('_') for sentence in text.split('. ')]

# train the Word2Vec model
model = Word2Vec(tokenized_text, vector_size=100, min_count=1, workers=4)

# Save the trained model
model.save("word2vec_properties.model")

# Update the hashed columns with the Word2Vec embeddings
embedding_data = pd.DataFrame()
for col in hashed_columns:
    embeddings = np.vstack(data[col].astype(str).fillna('').apply(lambda x: model.wv[x]))
    temp_df = pd.DataFrame(embeddings, columns=[f"{col}_embed_{i}" for i in range(embeddings.shape[1])])
    embedding_data = pd.concat([embedding_data, temp_df], axis=1)

data = pd.concat([data, embedding_data], axis=1)
data = data.drop(columns=hashed_columns)

# sort columns alphabetically 
data.sort_index(axis=1, inplace=True)

# End of preprocessing

# Convert 'date_of_birth' to a numeric format and fill invalid values with NaN
data['date_of_birth'] = pd.to_numeric(data['date_of_birth'], errors='coerce')

# Impute missing values for numerical columns with the mean value
numerical_columns = ['street_number', 'date_of_birth']
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Scaling numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Add an index column to the data
data['index_col'] = data.index

# Create a new column called "original_id" to indicate the corresponding original record ID
data['original_id'] = data['rec_id'].apply(lambda x: x.replace("-dup-0", "-org") if "-dup-" in x else x)

# printing this should show something like this
#           rec_id  culture  given_name  surname  street_number  address_1  state  date_of_birth  phone_number  national_identifier  blocking_number  sex  address_2  match  index_col original_id
#0       rec-0-org        0        -1.0      1.0      -0.578839       -1.0      4       1.062664          -1.0                 -1.0        -0.408013    2        1.0      0          0   rec-0-org

# Create a list of duplicate and non-duplicate indices
duplicate_indices = []
non_duplicate_indices = []

for original_id, duplicate_list in originals.items():
    original_id_index = data[data['original_id'] == original_id]['index_col'].values[0]
    non_duplicate_indices.append(original_id_index)
    duplicate_indices.extend(duplicate_list)

# Randomly select non-duplicate indices to match the number of duplicates
np.random.seed(42)  # Set a seed for reproducibility
non_duplicate_indices = np.random.choice(non_duplicate_indices, len(duplicate_indices), replace=False)

# Combine the selected duplicate and non-duplicate indices
selected_indices = np.concatenate((duplicate_indices, non_duplicate_indices))

# Create a balanced dataset
balanced_data = data.loc[selected_indices]

# Shuffle the balanced dataset
random_state = 42
balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Remove 'rec_id' and 'blocking_number' columns
balanced_data = balanced_data.drop(columns=['rec_id', 'blocking_number', 'original_id'])

# sort columns alphabetically 
balanced_data.sort_index(axis=1, inplace=True)

# Split the dataset into features and labels
X = balanced_data.drop(columns=['match'])
y = balanced_data['match']

# Use StratifiedKFold for cross-validation and model training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = None, None, None, None
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if len(np.unique(y_train)) >= 2:
        break

#########################################################################################
# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf.fit(X_train, y_train)

# Train the LinearSVC
svm = LinearSVC(random_state=random_state)
svm.fit(X_train, y_train)

# Train the GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state=random_state)
gbt.fit(X_train, y_train)

# Predict the labels for the test set using each model
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_gbt = gbt.predict(X_test)

# Evaluate the models
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gbt = accuracy_score(y_test, y_pred_gbt)

print("RandomForestClassifier Accuracy:", accuracy_rf)
print("LinearSVC Accuracy:", accuracy_svm)
print("GradientBoostingClassifier Accuracy:", accuracy_gbt)

##################################################################
# Test the models on the test set

# Define the new record
# this record should match
# these records need to match exactly in the order of properties
# new_record = {
#     "rec_id": "rec-1006",
#     "culture": "eng",
#     "sex": "f",
#     "given_name": "xxxxx",
#     "surname": "xxxx lr",
#     "street_number": 89,
#     "address_1": "xxxxx place",
#     "address_2": "",
#     "state": "qld",
#     "date_of_birth": "777777777777777",
#     "phone_number": "77777",
#     "national_identifier": "777777",
#     "index_col": 1006
# }
# should match
# new_record = {
#     "rec_id": "rec-3000",
#     "culture": "eng",
#     "sex": "m",
#     "given_name": "coiar",
#     "surname": "lukeson",
#     "street_number": 89,
#     "address_1": "jacobs street",
#     "address_2": "rose gardens touri",
#     "state": "vic",
#     "date_of_birth": "19791215",
#     "phone_number": "8418369",
#     "national_identifier": "86326798",
#     "index_col": 3000
# }
# should not match
new_record = {
    "rec_id": "rec-3010",
    "culture": "eng",
    "sex": "m",
    "given_name": "Dane",
    "surname": "Hollenbach",
    "street_number": 77,
    "address_1": "church street",
    "address_2": "reigate",
    "state": "vic",
    "date_of_birth": "19901122",
    "phone_number": "022269553",
    "national_identifier": "12345678",
    "index_col": 3010
}

# preprocess the new record
new_data = pd.DataFrame([new_record])

# Remove the 'rec_id' column from the new_data DataFrame
new_data = new_data.drop(columns=["rec_id"])

# Clean the data
# Fill missing values for 'culture', 'sex', and 'state' with a default value
default_values = {'culture': 'unknown', 'sex': 'unknown', 'state': 'unknown'}
new_data.fillna(value=default_values, inplace=True)

# Remove extra spaces from string columns
string_columns = ['given_name', 'surname', 'address_1', 'address_2', 'phone_number', 'national_identifier']
for col in string_columns:
    new_data[col] = new_data[col].astype(str).str.strip()

# Label encoding for 'culture', 'sex', and 'state'
label_encoder_columns = ['culture', 'sex', 'state']
for col in label_encoder_columns:
    label_encoder = LabelEncoder()
    new_data[col] = label_encoder.fit_transform(new_data[col].astype(str))

# Feature hashing for string columns
hasher = FeatureHasher(n_features=1000000, input_type='string')
hashed_columns = ['given_name', 'surname', 'address_1', 'address_2', 'phone_number', 'national_identifier']

# Load the trained model
# Use the same word2vec model as the one used for training
trained_model = Word2Vec.load("word2vec_properties.model")

# Use Word2Vec to encode string columns
text_new = ''

for col in hashed_columns:
    text_new += '_'.join(new_data[col].astype(str).fillna('')) + '_'

tokenized_text_new = [sentence.split('_') for sentence in text_new.split('. ')]

# Update the vocabulary with new data
trained_model.build_vocab(tokenized_text_new, update=True)

# Train the model on the new data
trained_model.train(tokenized_text_new, total_examples=len(tokenized_text_new), epochs=model.epochs)

embedding_new_data = pd.DataFrame()
for col in hashed_columns:
    embeddings = np.vstack(new_data[col].astype(str).fillna('').apply(lambda x: trained_model.wv[x]))
    temp_df = pd.DataFrame(embeddings, columns=[f"{col}_embed_{i}" for i in range(embeddings.shape[1])])
    embedding_new_data = pd.concat([embedding_new_data, temp_df], axis=1)

new_data = pd.concat([new_data, embedding_new_data], axis=1)
new_data = new_data.drop(columns=hashed_columns)

# sort columns alphabetically 
new_data.sort_index(axis=1, inplace=True)

trained_model.save("word2vec_properties_updated.model")

# Convert 'date_of_birth' to a numeric format and fill invalid values with NaN
new_data['date_of_birth'] = pd.to_numeric(new_data['date_of_birth'], errors='coerce')

# Impute missing values for numerical columns with the mean value
numerical_columns = ['street_number', 'date_of_birth']
imputer = SimpleImputer(strategy='mean')
new_data[numerical_columns] = imputer.fit_transform(new_data[numerical_columns])

# Scaling numerical columns
scaler = StandardScaler()
new_data[numerical_columns] = scaler.fit_transform(new_data[numerical_columns])
# End clean the data

new_data[numerical_columns] = imputer.transform(new_data[numerical_columns])
new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

# predict the probability of the new record being a duplicate
rf_probability = rf.predict_proba(new_data)[:, 1]
svm_probability = svm.decision_function(new_data)
gbt_probability = gbt.predict_proba(new_data)[:, 1]

# print the probability
print('#########################################')
print("RandomForestClassifier Probability of Match:", rf_probability)
print("LinearSVC Probability of Match:", svm_probability)
print("GradientBoostingClassifier Probability of Match:", gbt_probability)

similarities = cosine_similarity(X_train, new_data)
most_similar_index = similarities.argmax()

print('#########################################')
# Set the match probability thresholds
match_threshold = 0.85
potential_match_threshold = 0.65

# Function to classify the new record based on the probability, the defined thresholds, and the model name
def classify_and_print_matched_record(prob, model_name):
    if prob >= match_threshold:
        print(f"Record match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")
        print(f"Matched with record:\n{X_train.iloc[most_similar_index]}\n")
    elif prob >= potential_match_threshold:
        print(f"Potential record match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")
        print(f"Matched with record:\n{X_train.iloc[most_similar_index]}\n")
    else:
        print(f"No match found for rec_id {new_record['rec_id']} with {model_name} probability: {prob}")

# Classify the new record and print the matched record based on the probabilities and the defined thresholds for each model
print(f"\nRandom Forests Prob {rf_probability}:")
classify_and_print_matched_record(rf_probability[0], "Random Forests")
print(f"\nLinear SVM  {svm_probability}:")
classify_and_print_matched_record(svm_probability[0], "Linear SVM")
print(f"\nGradient Boosted Trees  {gbt_probability}:")
classify_and_print_matched_record(gbt_probability[0], "Gradient Boosted Trees")

# print of the columns in a dataframe
print(X_train.columns)