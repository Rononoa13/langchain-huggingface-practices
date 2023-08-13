from transformers import pipeline

# Create sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
# # Analyze sentiment for a given text input
text = "I love this product! It's amazing."
sentiment = sentiment_analyzer(text)
# # Print sentiment result
print("Text: ", text)
print("Sentiment Label: ", sentiment[0]['label'])
print("Sentiment Score: ", sentiment[0]['score'])

