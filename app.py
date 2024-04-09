from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the trained model from a file
with open('trained_model.joblib', 'rb') as f:
    model = joblib.load(f)

@app.route('/score', methods=['POST'])
def score_text():
    text = request.form['text']
    threshold = 0.5
    
    prediction, propensity = score(text, model, threshold)
    
    return jsonify({
        'prediction': prediction,
        'propensity': propensity
    })

if __name__ == '__main__':
    app.run(debug=True)