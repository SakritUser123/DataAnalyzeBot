from flask import Flask, render_template,request

# Encode input text (prompt)

from tensorflow.keras.layers import TextVectorization
with open('emotions.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
X_test = ["hello"]
vectorizer = TextVectorization(max_tokens=50,output_mode='int')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])  # Both GET and POST are allowed here
def home():
    user_input = None
    
    # Handle POST request
    if request.method == "POST":
        user_input = request.form.get("user_input")
        print(f"User input: {user_input}")
        X_test = user_input
        vectorizer.adapt(X_test)
        vectorized_texts = vectorizer(X_test)
        
        predictions = loaded_model.predict(vectorized_texts)
        print(predictions)
        threshold = 0.25
        bin = (predictions >= threshold).astype(int)
        labels_text = ['positive' if label == 1 else 'negative' for label in bin.flatten()]
        print(labels_text)

        

    # Generate text from the model
        

    # Decode and print the output
        
        # For debugging: print user input to terminal
        return jsonify({"generated_text": labels_text})
    return render_template("index.html",labels_text=labels_text)

if __name__ == "__main__":
    app.run(debug=True)
