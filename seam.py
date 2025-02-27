from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Input text from user
inputtext = input("Enter what you want to do with the data: ")

# Tokenizing input text
inputs = tokenizer(inputtext, return_tensors="pt", truncation=True)

# Check if the input text was tokenized properly
if inputs['input_ids'].size(1) == 0:
    print("Error: Tokenized input is empty.")
else:
    # Generate output based on tokenized input
    outputs = model.generate(inputs['input_ids'], max_length=50,attention_mask=inputs['attention_mask'])

    # Decode the generated token IDs back to text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the result
    print(decoded_output)
