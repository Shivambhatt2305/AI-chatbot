from flask import Flask, request, render_template, jsonify
import os
import google.generativeai as genai
import markdown
import time

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure Gemini API
API_KEY = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = API_KEY
genai.configure(api_key=API_KEY)

# Initialize model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model = None

# Store chat history in memory
chat_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    if user_input.lower() == 'exit':
        return jsonify({"response": "Chat session ended."})
    
    if not model:
        return jsonify({"error": "Model not initialized. Check API key and model availability."})
    
    try:
        # Generate response with streaming
        api_response = model.generate_content(user_input)
        response_text = api_response.text
        
        # Add to chat history
        chat_entry = {
            'question': user_input,
            'response': response_text,
            'timestamp': time.strftime('%H:%M:%S')
        }
        chat_history.append(chat_entry)
        
        return jsonify({
            "response": response_text,
            "timestamp": chat_entry['timestamp']
        })
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify({"error": error_message})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"status": "success", "message": "Chat history cleared"})

if __name__ == '__main__':
    app.run(debug=True)
