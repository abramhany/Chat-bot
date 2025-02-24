import random
import pickle
import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words =pickle.load(open('words.pkl','rb'))

classes = pickle.load(open('classes.pkl','rb'))

model_path = 'chatbot_model.h5'

model = tf.keras.models.load_model(model_path)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i]= 1
    return np.array(bag)           

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print(f"Predicted intents: {return_list}")  # Debugging line
    return return_list

def get_response(intents_list, intents_json, input_message):
    if not intents_list:
        return "I'm not sure how to respond to that."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    message = input('').lower()
    ints = predict_class(message)
    res = get_response(ints, intents, message)
    print(res)