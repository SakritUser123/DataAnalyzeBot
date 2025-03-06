from flask import Flask, render_template,request
app = Flask(__name__)
from flask import Flask, render_template,request
import pickle
# Encode input text (prompt)

from tensorflow.keras.layers import TextVectorization
with open('emotions.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
X_test = ["hello"]
vectorizer = TextVectorization(max_tokens=50,output_mode='int')
@app.route("/", methods=["GET", "POST"])  # Both GET and POST are allowed here
def home():
    user_input = None
    
    # Handle POST request
    if request.method == "POST":
        user_input = request.form.get("user_input")
        print(f"User input: {user_input}")
        X_test = user_input
        vectorizer.adapt(user_input)
        vectorized_texts = vectorizer(user_input)
        
        predictions = loaded_model.predict(vectorized_texts)
        print(predictions)
        threshold = 0.25
        bin = (predictions >= threshold).astype(int)
        labels_text = ['positive' if label == 1 else 'negative' for label in bin.flatten()]
        print(labels_text)

if __name__ == "__main__":
    app.run(debug=True)
