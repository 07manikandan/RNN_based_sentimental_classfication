import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SimpleRNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv('data.csv')
df = df.dropna()
X = df['Sentence']
y = df['Sentiment']

# Reset index
message = X.reset_index(drop=True)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# SpaCy preprocessing function
def preprocess_text_spacy(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\$\w+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text.lower())
    cleaned_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(cleaned_tokens)

# Apply preprocessing
corpus = [preprocess_text_spacy(title) for title in message]

# One-hot encoding
voc_size = 5000
onehot = [one_hot(words, voc_size) for words in corpus]

# Padding sequences
length = 40
embedded = pad_sequences(onehot, padding='pre', maxlen=length)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to NumPy arrays
X_final = np.array(embedded)
y_final = np.array(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, stratify=y_final
)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Build RNN model
embedding_vector_feature = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_feature, input_length=length))
model.add(SimpleRNN(128, activation='relu', return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(y_final)), activation="softmax"))  # output classes = unique labels

model.build(input_shape=(None, length))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Prediction function for new text
def predict_class(text):
    cleaned_text = preprocess_text_spacy(text)
    onehot_encoded = one_hot(cleaned_text, voc_size)
    padded = pad_sequences([onehot_encoded], padding='pre', maxlen=length)
    pred_prob = model.predict(padded)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    label = label_encoder.inverse_transform([pred_class])[0]

    print(f"\nInput Text: {text}")
    print(f"Cleaned Tokens: {cleaned_text}")
    print(f"Predicted Class: {pred_class} ({label})")
    print(f"Confidence: {np.max(pred_prob):.2f}")

# Interactive prediction loop
while True:
    input_text = input("\nEnter a review (or type 'exit' to stop): ")
    if input_text.lower() == 'exit':
        break
    predict_class(input_text)
