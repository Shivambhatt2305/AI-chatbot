rom flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_session import Session
from flask import Flask, request, render_template, jsonify
import os
import google.generativeai as genai
from datetime import datetime
import bleach
import markdown
from markupsafe import Markup
import functools
import logging
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
Session(app)

# Response cache with TTL
response_cache = {}
CACHE_TTL = 3600  # 1 hour cache lifetime
app = Flask(__name__, static_folder='static')

# Configure Gemini API
try:
    API_KEY = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Your provided API key
    os.environ['GOOGLE_API_KEY'] = API_KEY
    genai.configure(api_key=API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Error configuring API: {e}")
API_KEY = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = API_KEY
genai.configure(api_key=API_KEY)

# Initialize model with safety settings
# Initialize model
try:
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        generation_config=generation_config,
        safety_settings=safety_settings
        'gemini-1.5-flash',
        system_instruction="""You are a helpful AI assistant. You maintain context from previous messages in the conversation. 
        When users ask follow-up questions like "explain in detail", "give me more info", "elaborate", etc., 
        refer back to the previous topics discussed in the conversation."""
)
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    print(f"Error initializing model: {str(e)}")
model = None

# Cache decorator
def cache_response(func):
    @functools.wraps(func)
    def wrapper(prompt, *args, **kwargs):
        # Create a cache key from the prompt
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check if response is in cache and not expired
        current_time = time.time()
        if cache_key in response_cache:
            cached_response, timestamp = response_cache[cache_key]
            if current_time - timestamp < CACHE_TTL:
                logger.info(f"Cache hit for prompt: {prompt[:30]}...")
                return cached_response
        
        # Generate new response
        result = func(prompt, *args, **kwargs)
        
        # Cache the result
        response_cache[cache_key] = (result, current_time)
        
        # Clean old cache entries
        clean_cache()
        
        return result
    return wrapper

def clean_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [k for k, (_, t) in response_cache.items() if current_time - t > CACHE_TTL]
    for key in expired_keys:
        del response_cache[key]
# Store chat sessions
chat_sessions = {}
current_session_id = "default"
chat_history = []

# Session-based chat history
def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def save_chat_history(history):
    session['chat_history'] = history
    session.modified = True

# Sanitize user input
def sanitize_input(text):
    return bleach.clean(text, tags=[], strip=True)
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

@cache_response
def generate_ai_response(prompt):
    """Generate response from AI with caching"""
    try:
        # Enhanced prompt for better responses
        system_prompt = """You are a helpful, accurate, and concise assistant. 
        Format your responses in markdown for readability. 
        Include code examples when relevant. 
        Be direct and to the point while remaining helpful."""
        
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"{system_prompt}\n\nUser question: {prompt}"
        )
        
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"
# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET'])
def home():
    if request.method == 'POST':
        user_input = sanitize_input(request.form.get('user_input', '').strip())

        # Handle empty input
        if not user_input:
            flash("Please enter a question.", "error")
            return redirect(url_for('home'))

        # Handle 'exit' command
        if user_input.lower() == 'exit':
            session.pop('chat_history', None)
            flash("Chat session ended.", "info")
            return redirect(url_for('home'))

        # Check if model is initialized
        if not model:
            flash("Error: Model not initialized. Check API configuration.", "error")
            return redirect(url_for('home'))

        try:
            # Check for duplicate question
            chat_history = get_chat_history()
            if chat_history and chat_history[-1].get('question') == user_input:
                flash("This question was already answered. See below.", "warning")
                return redirect(url_for('home'))

            # Generate response with caching
            start_time = time.time()
            response_text = generate_ai_response(user_input)
            response_time = time.time() - start_time
            
            # Convert markdown to HTML
            response_html = Markup(markdown.markdown(
                response_text,
                extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
            ))

            # Add to chat history
            chat_history.append({
                'question': user_input,
                'response': response_html,
                'response_time': f"{response_time:.2f}s",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Limit history to prevent session bloat
            save_chat_history(chat_history[-50:])  # Keep last 50 entries

            flash(f"Response generated in {response_time:.2f} seconds", "success")

        except Exception as e:
            logger.error(f"Error in chat route: {e}")
            flash(f"Error generating response: {e}", "error")

        return redirect(url_for('home'))

    # For GET requests
    return render_template('index.html', chat_history=get_chat_history())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('chat_history', None)
    flash("Chat history cleared.", "info")
    return redirect(url_for('home'))
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

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for AJAX chat requests"""
@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'success': False, 'error': 'No message provided'}), 400
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    user_input = sanitize_input(data['message'])
    if user_input.lower() == 'exit':
        return jsonify({"response": "Chat session ended."})

if not model:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 500
        return jsonify({"error": "Model not initialized. Check API key and model availability."})

try:
        # Generate response
        response_text = generate_ai_response(user_input)
        response_html = markdown.markdown(
            response_text,
            extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
        )
        # Get or create chat session
        session = conversation_manager.get_or_create_session(current_session_id)
        chat_session = session['chat']
        
        # Send message with full conversation context
        response = chat_session.send_message(user_input)
        response_text = response.text
        
        # Add to conversation history
        conversation_manager.add_message(current_session_id, user_input, response_text)

        # Add to chat history
        chat_history = get_chat_history()
        # Add to global chat history for template
chat_history.append({
'question': user_input,
            'response': Markup(response_html),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            'response': response_text,
            'timestamp': time.strftime('%H:%M:%S')
})
        save_chat_history(chat_history[-50:])

return jsonify({
            'success': True,
            'response': response_text,
            'response_html': response_html,
            'timestamp': datetime.now().isoformat()
            "response": response_text,
            "timestamp": time.strftime('%H:%M:%S')
})

except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
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
        'status': 'healthy',
        'model_initialized': model is not None,
        'api_key_configured': 'GOOGLE_API_KEY' in os.environ,
        'cache_size': len(response_cache),
        'timestamp': datetime.now().isoformat()
        "session_history": session_history,
        "chat_history": chat_history,
        "session_id": current_session_id
})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True)
