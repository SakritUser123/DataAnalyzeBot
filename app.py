from flask import Flask, render_template, request
import pickle
from tensorflow.keras.layers import TextVectorization

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('emotions.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Initialize the vectorizer
vectorizer = TextVectorization(max_tokens=50, output_mode='int')

@app.route("/", methods=["GET", "POST"])  # Both GET and POST are allowed here
def home():
    labels_text = []

    if request.method == "POST":
        # Get user input from the form
        user_input = request.form.get("user_input")
        print(f"User input: {user_input}")

        if not user_input:  # Check if user_input is empty or None
            return "Please enter some text to analyze."

        # Adapt the vectorizer to the user input (wrap it in a list)
        user_input_list = [user_input]  # TextVectorization expects an iterable
        vectorizer.adapt(user_input_list)  # Adapt to the input list

        # Vectorize the input text
        vectorized_texts = vectorizer(user_input_list)
        print(f"Vectorized input: {vectorized_texts}")

        # Predict with the loaded model
        predictions = loaded_model.predict(vectorized_texts)
        print(f"Predictions: {predictions}")

        # Apply threshold and generate binary labels
        threshold = 0.25
        bin_labels = (predictions >= threshold).astype(int)
        labels_text = ['positive' if label == 1 else 'negative' for label in bin_labels.flatten()]

        print(f"Labels: {labels_text}")

        # Return the prediction result to the user
        return render_template('index.html', user_input=user_input, labels_text=labels_text)

    # Render the home page with a form (GET request)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
