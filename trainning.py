import random
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.utils.class_weight import compute_class_weight

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess data
intents = json.loads(open(r'intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open(r'words.pkl', 'wb'))
pickle.dump(classes, open(r'classes.pkl', 'wb'))

# Add after data preprocessing, before model training
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and lemma.name() not in synonyms:
                synonyms.append(lemma.name())
    return synonyms[:2]  # Return up to 2 synonyms

# Augment training data
augmented_documents = documents.copy()
for doc, tag in documents:
    augmented_pattern = []
    for word in doc:
        synonyms = get_synonyms(word)
        if synonyms:
            # Add a pattern with one word replaced by its synonym
            for syn in synonyms:
                new_pattern = doc.copy()
                new_pattern[new_pattern.index(word)] = syn
                augmented_documents.append((new_pattern, tag))

documents = augmented_documents

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Split features and labels
X = training[:, :len(words)]
y = training[:, len(words):]

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model with improved architecture
model = tf.keras.Sequential([
    # Input layer with more units and batch normalization
    tf.keras.layers.Dense(512, input_shape=(len(words),), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Hidden layers with decreasing units
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Output layer
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile with modified parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modified early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor accuracy instead of loss
    patience=30,            # Increased patience
    restore_best_weights=True,
    min_delta=0.001        # Minimum change to qualify as an improvement
)

# Add learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.0001
)

# Get the true labels (convert one-hot encoded y_train to class indices)
y_train_labels = np.argmax(y_train, axis=1)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

# Convert to dictionary
class_weight_dict = dict(zip(np.unique(y_train_labels), class_weights))

# Train with modified parameters
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,            
    batch_size=16,        
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict  # Use the computed class weights
)

# Print final metrics
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f'Final Training Accuracy: {final_train_accuracy:.4f}')
print(f'Final Validation Accuracy: {final_val_accuracy:.4f}')

# Save the model
model.save('chatbot_model.h5')
print('Training completed and model saved!')

