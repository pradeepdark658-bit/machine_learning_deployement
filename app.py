from flask import Flask, request, jsonify, Flask
import joblib
import re

app = Flask(__name__)

# Load the saved model components
model = joblib.load("logistic_regression_model.joblib")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")


# Function to clean text
# Must be the same preprocessing used during training
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters and spaces
    return text


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether JSON data is provided
    if not request.is_json:
        return jsonify({
            "error": "Request must contain JSON data."
        }), 400

    data = request.get_json()

    # Check whether 'text' field exists
    if "text" not in data:
        return jsonify({
            "error": "Please provide a 'text' field in the JSON request."
        }), 400

    new_article_text = data["text"]

    # Check if text is empty
    if not new_article_text.strip():
        return jsonify({
            "error": "Text cannot be empty."
        }), 400

    # Preprocess the new text
    cleaned_text = clean_text(new_article_text)

    # Convert text into TF-IDF features
    new_article_tfidf = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    # Make prediction
    prediction_numerical = model.predict(
        new_article_tfidf
    )[0]

    # Convert numerical prediction back to original label
    prediction_label = label_encoder.inverse_transform(
        [prediction_numerical]
    )[0]

    # Return prediction as JSON
    return jsonify({
        "prediction": prediction_label
    })


# Run Flask application
if __name__ == "__main__":
    print("Flask API is running...")
    print("API endpoint: http://127.0.0.1:5000/predict")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
