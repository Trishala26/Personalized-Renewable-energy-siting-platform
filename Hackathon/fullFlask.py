from flask import Flask, render_template, request, redirect, url_for,jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)

gb_model = pickle.load(open("knn.pkl", 'rb'))

# Load necessary data and models
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r"C:\Users\trilo\OneDrive\Desktop\Trishala\ADShackathon\Hackathon\EnergyIntents.json").read())
words = pickle.load(open(r"C:\Users\trilo\OneDrive\Desktop\Trishala\ADShackathon\Hackathon\words.pkl", 'rb'))
classes = pickle.load(open(r"C:\Users\trilo\OneDrive\Desktop\Trishala\ADShackathon\Hackathon\classes.pkl", 'rb'))
model = load_model(r"C:\Users\trilo\OneDrive\Desktop\Trishala\ADShackathon\Hackathon\chatbot_model.h5")

# Define helper functions for chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/welcome.html')
def welcome():
    return render_template('welcome.html')

@app.route('/index.html')
def home_index():
    return render_template('index.html')

@app.route('/back.html')
def home_back():
    return render_template('home.html')

@app.route('/home.html')
def home_home():
    return render_template('home.html')

@app.route('/chat.html')
def home_chat():
    return render_template('chat.html')


@app.route('/get')
def get_bot_response():
    user_input = request.args.get('msg')
    ints = predict_class(user_input)
    response = get_response(ints, intents)
    return response

@app.route('/map.html')
def map():
    return render_template('map.html')

@app.route('/predict_gb', methods=['POST'])
def predict_gb():
    try:
        # Get the data from the form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        energy_need = float(request.form['energy'])
        budget = float(request.form['budget'])
        square_feet = float(request.form['squarefeet'])
        data = np.array([[latitude, longitude, energy_need, budget, square_feet]])
        prediction = gb_model.predict(data)

               # Pass the predicted label to the template
        return render_template('prediction (1).html', prediction=prediction)
    except Exception as e:
        return render_template('prediction (1).html', error=str(e))
if __name__ == '__main__':
    app.run(debug=True)