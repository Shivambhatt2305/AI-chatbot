from flask import Flask, request, render_template, redirect, url_for, flash
import os
import google.generativeai as genai
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key"  # Required for flash messages

# Configure Gemini API
try:
    os.environ['GOOGLE_API_KEY'] = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Replace with your actual Gemini API key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except Exception as e:
    print(f"Error configuring API: {e}")

# Initialize model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None

# Store chat history in memory
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()

        # Handle empty input
        if not user_input:
            flash("Please enter a question.")
            return redirect(url_for('home'))

        # Handle 'exit' command
        if user_input.lower() == 'exit':
            chat_history.clear()
            flash("Chat session ended.")
            return redirect(url_for('home'))

        # Check if model is initialized
        if not model:
            flash("Error: Model not initialized. Check API configuration.")
            return redirect(url_for('home'))

        try:
            # Check for duplicate question in recent history
            if chat_history and chat_history[-1].get('question') == user_input:
                flash("This question was already answered. See the chat history below.")
                return redirect(url_for('home'))

            # Generate content without streaming for reliability
            api_response = model.generate_content(user_input)
            
            # Extract response text
            response_text = api_response.text.strip() if api_response.text else "No response generated."
            
            # Add to chat history
            chat_history.append({
                'question': user_input,
                'response': response_text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            flash("Response generated successfully!")

        except Exception as e:
            flash(f"Error generating response: {e}")
        
        return redirect(url_for('home'))

    # For GET requests, render the template
    return render_template('index.html', user_input="", chat_history=chat_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    chat_history.clear()
    flash("Chat history cleared.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
