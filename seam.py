from flask import Flask, render_template,request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input text (prompt)


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])  # Both GET and POST are allowed here
def home():
    user_input = None
    
    # Handle POST request
    if request.method == "POST":
        user_input = request.form.get("user_input")
        print(f"User input: {user_input}")
        input_text = user_input
        inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text from the model
        output = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)

    # Decode and print the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # For debugging: print user input to terminal
        return jsonify({"generated_text": generated_text})
    return render_template("index.html",generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
