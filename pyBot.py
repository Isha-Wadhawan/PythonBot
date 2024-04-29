
import tkinter as tk
from tkinter import scrolledtext, END
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

class ChatbotApp:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        self.chat_history = scrolledtext.ScrolledText(master, width=50, height=20)
        self.chat_history.grid(row=0, column=0, columnspan=2)

        self.user_input = tk.Entry(master, width=40)
        self.user_input.grid(row=1, column=0, padx=5, pady=5)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)

        # Step 1: Parse the JSON file
        with open('intents.json', 'r') as file:
            self.data = json.load(file)

        # Step 2: Preprocess the Data
        patterns = []
        responses = []
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                responses.append(intent['tag'])

        # Step 3: Vectorize the Data
        self.tfidf_vectorizer = TfidfVectorizer()
        self.X = self.tfidf_vectorizer.fit_transform(patterns)

        # Step 4: Train a Model
        self.classifier = LinearSVC()
        self.classifier.fit(self.X, responses)

    def send_message(self):
        user_input_text = self.user_input.get()
        self.add_chat_entry("You: " + user_input_text)

        # Step 5: Predict Response
        user_input_vectorized = self.tfidf_vectorizer.transform([user_input_text])
        predicted_intent = self.classifier.predict(user_input_vectorized)[0]
        for intent in self.data['intents']:
            if intent['tag'] == predicted_intent:
                response = random.choice(intent['responses'])
                self.add_chat_entry("Bot: " + response)
                break

        self.user_input.delete(0, END)

    def add_chat_entry(self, message):
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.see(tk.END)

root = tk.Tk()
app = ChatbotApp(root)
root.mainloop()

