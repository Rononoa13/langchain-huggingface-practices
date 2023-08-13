from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def generate_prompt():
    if request.method == "POST":
        prompt = request.form["prompt"]
        generated_text = generate_text(prompt)
        return render_template("index.html", prompt=prompt, generated_text=generated_text)
    return render_template("index.html")

def generate_text(prompt):
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return generated_text


if __name__ == "__main__":
    app.run(debug=True)
