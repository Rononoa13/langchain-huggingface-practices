from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = Flask(__name__)

messages = []

@app.route("/", methods=["GET", "POST"])
def chat():
    global messages
    if request.method == "POST":
        user_input = request.form["user-input"]
        bot_response = generate_response(user_input)
        messages.append({"user": user_input, "bot": bot_response})
        return render_template("index.html", user_input=user_input, bot_response=bot_response, messages=messages)
    return render_template("index.html")

# generates the bot's response using the language model:
def generate_response(user_input):
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    bot_response = text_generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
    return bot_response

if __name__ == "__main__":
    app.run(debug=True)
