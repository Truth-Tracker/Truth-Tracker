from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the brain (model) and the translator (vectorizer)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news_text']
        # 1. Transform the input text using the vectorizer
        data = [message]
        vect = vectorizer.transform(data).toarray()
        # 2. Predict using the model
        prediction = model.predict(vect)
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)