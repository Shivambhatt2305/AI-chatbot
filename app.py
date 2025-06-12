from flask import Flask, request, render_template, jsonify
import os
import google.generativeai as genai
import time

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure Gemini API
API_KEY = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = API_KEY
genai.configure(api_key=API_KEY)

# Initialize model
try:
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction="""You are a helpful AI assistant. You maintain context from previous messages in the conversation. 
        When users ask follow-up questions like "explain in detail", "give me more info", "elaborate", etc., 
        refer back to the previous topics discussed in the conversation."""
    )
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model = None

# Store chat sessions
chat_sessions = {}
current_session_id = "default"
chat_history = []

class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'chat': model.start_chat(history=[]),
                'history': []
            }
        return self.sessions[session_id]
    
    def add_message(self, session_id, user_message, ai_response):
        session = self.get_or_create_session(session_id)
        session['history'].append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': time.strftime('%H:%M:%S')
        })
    
    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_history(self, session_id):
        if session_id in self.sessions:
            return self.sessions[session_id]['history']
        return []

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route('/', methods=['GET'])
def home():
    session_history = conversation_manager.get_session_history(current_session_id)
    # Convert to the format expected by the template
    formatted_history = []
    for item in session_history:
        formatted_history.append({
            'question': item['user'],
            'response': item['ai'],
            'timestamp': item['timestamp']
        })
    return render_template('index.html', chat_history=formatted_history)

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
        # Get or create chat session
        session = conversation_manager.get_or_create_session(current_session_id)
        chat_session = session['chat']
        
        # Send message with full conversation context
        response = chat_session.send_message(user_input)
        response_text = response.text
        
        # Add to conversation history
        conversation_manager.add_message(current_session_id, user_input, response_text)
        
        # Add to global chat history for template
        chat_history.append({
            'question': user_input,
            'response': response_text,
            'timestamp': time.strftime('%H:%M:%S')
        })
        
        return jsonify({
            "response": response_text,
            "timestamp": time.strftime('%H:%M:%S')
        })
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"Detailed error: {e}")  # For debugging
        
        # Try to recover by creating a new session
        try:
            conversation_manager.clear_session(current_session_id)
            session = conversation_manager.get_or_create_session(current_session_id)
            chat_session = session['chat']
            
            # Add context from recent history
            recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
            context_prompt = "Previous conversation context:\n"
            for item in recent_history:
                context_prompt += f"User: {item['question']}\nAI: {item['response']}\n\n"
            context_prompt += f"Current question: {user_input}"
            
            response = chat_session.send_message(context_prompt)
            response_text = response.text
            
            conversation_manager.add_message(current_session_id, user_input, response_text)
            chat_history.append({
                'question': user_input,
                'response': response_text,
                'timestamp': time.strftime('%H:%M:%S')
            })
            
            return jsonify({
                "response": response_text,
                "timestamp": time.strftime('%H:%M:%S')
            })
            
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")
            return jsonify({"error": f"Error: {str(e)}"})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    conversation_manager.clear_session(current_session_id)
    return jsonify({"status": "success", "message": "Chat history cleared"})

@app.route('/get_context', methods=['GET'])
def get_context():
    """Debug endpoint to see conversation context"""
    session_history = conversation_manager.get_session_history(current_session_id)
    return jsonify({
        "session_history": session_history,
        "chat_history": chat_history,
        "session_id": current_session_id
    })

if __name__ == '__main__':
    app.run(debug=True)
