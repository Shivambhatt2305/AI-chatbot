from flask import Flask, request, render_template, redirect, url_for, session
import os
import google.generativeai as genai
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Configure Gemini API
try:
    os.environ['GOOGLE_API_KEY'] = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Replace with your actual API key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except Exception as e:
    print(f"Error configuring API: {str(e)}")

# Initialize model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    user_input = ""

    # Get chat history from session (user-specific)
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()

        if user_input.lower() == 'exit':
            session['chat_history'] = []
            return render_template('index.html', 
                                   response="Chat session ended.", 
                                   user_input="", 
                                   chat_history=[])

        if not model:
            response = "Error: Model not initialized. Please check API configuration."
        else:
            try:
                api_response = model.generate_content(user_input, stream=True)

                response_text = ""
                for chunk in api_response:
                    if chunk.text:
                        response_text += chunk.text

                response = response_text

                if response:
                    chat_history.append({
                        'question': user_input,
                        'response': response,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    session['chat_history'] = chat_history  # Save updated history to session

            except Exception as e:
                response = f"Error generating response: {str(e)}"

    return render_template('index.html', 
                           response=response, 
                           user_input=user_input, 
                           chat_history=chat_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    return redirect(url_for('home'))

# Do NOT use debug=True when deploying
if __name__ == '__main__':
    app.run()
