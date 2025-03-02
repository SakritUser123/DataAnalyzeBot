from flask import Flask, render_template,request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])  # Both GET and POST are allowed here
def home():
    user_input = None
    
    # Handle POST request
    if request.method == "POST":
        user_input = request.form.get("user_input")
        print(f"User input: {user_input}")  # For debugging: print user input to terminal
    
    return render_template("index.html", user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
