from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import pickle
import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data
try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = tf.keras.models.load_model('chatbot_model.h5')
except Exception as e:
    print(f"Error loading files: {str(e)}")
    raise

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
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
    return return_list

def get_response(intents_list, intents_json, input_message):
    if not intents_list:
        return "I'm not sure how to respond to that."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "I'm not sure how to respond to that."

# Test route
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "API is working!"})

# Main chatbot endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get message from request
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json()
                message = str(data.get('message', ''))
            else:
                message = str(request.form.get('message', ''))
        else:  # GET request
            message = str(request.args.get('message', ''))

        # Check if message is empty
        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Get prediction and response
        ints = predict_class(message)
        response = get_response(ints, intents, message)

        return jsonify({
            "response": response,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False) 