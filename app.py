from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load trained model and preprocessing files
model = joblib.load("logistic_regression_model.joblib")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    # Check JSON request
    if not request.is_json:
        return jsonify({
            "error": "Request must contain JSON data"
        }), 400

    data = request.get_json()

    # Check text field
    if "text" not in data:
        return jsonify({
            "error": "Please provide a 'text' field"
        }), 400

    article_text = data["text"]

    # Check empty text
    if not article_text.strip():
        return jsonify({
            "error": "Text cannot be empty"
        }), 400

    # Clean text
    cleaned_text = clean_text(article_text)

    # Convert text to TF-IDF
    text_tfidf = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    # Predict
    prediction_number = model.predict(
        text_tfidf
    )[0]

    # Convert prediction to original label
    prediction_label = label_encoder.inverse_transform(
        [prediction_number]
    )[0]

    return jsonify({
        "prediction": str(prediction_label)
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
