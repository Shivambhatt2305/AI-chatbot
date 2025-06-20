from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_session import Session
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

# Response cache with TTL - Modified to include context
response_cache = {}
CACHE_TTL = 3600  # 1 hour cache lifetime

# Configure Gemini API
try:
    API_KEY = "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I"  # Your provided API key
    os.environ['GOOGLE_API_KEY'] = API_KEY
    genai.configure(api_key=API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Error configuring API: {e}")

# Initialize model with safety settings
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
    )
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    model = None

# Cache decorator - Modified to include conversation context
def cache_response(func):
    @functools.wraps(func)
    def wrapper(prompt, chat_history=None, *args, **kwargs):
        # Create a cache key from the prompt and recent conversation context
        context_str = ""
        if chat_history and len(chat_history) > 0:
            # Include last 3 conversations for context in cache key
            recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
            context_str = str([(h.get('question', ''), h.get('response_text', '')) for h in recent_history])
        
        cache_key = hashlib.md5((prompt + context_str).encode()).hexdigest()
        
        # Check if response is in cache and not expired
        current_time = time.time()
        if cache_key in response_cache:
            cached_response, timestamp = response_cache[cache_key]
            if current_time - timestamp < CACHE_TTL:
                logger.info(f"Cache hit for prompt: {prompt[:30]}...")
                return cached_response
        
        # Generate new response
        result = func(prompt, chat_history, *args, **kwargs)
        
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

def format_conversation_history(chat_history, max_history=10):
    """Format conversation history for the AI model"""
    if not chat_history:
        return []
    
    # Get recent conversation history (limit to prevent token overflow)
    recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
    
    formatted_history = []
    for entry in recent_history:
        # Add user message
        formatted_history.append({
            "role": "user",
            "parts": [entry.get('question', '')]
        })
        
        # Add AI response (use raw text, not HTML)
        response_text = entry.get('response_text', '') or entry.get('response', '')
        if response_text:
            # If response is HTML markup, try to extract text
            if hasattr(response_text, '__html__'):
                response_text = str(response_text)
            # Remove HTML tags for AI context
            import re
            response_text = re.sub('<[^<]+?>', '', str(response_text))
            
            formatted_history.append({
                "role": "model",
                "parts": [response_text]
            })
    
    return formatted_history

@cache_response
def generate_ai_response(prompt, chat_history=None):
    """Generate response from AI with conversation context"""
    try:
        # Enhanced system prompt
        system_prompt = """You are a helpful, accurate, and concise assistant. 
        You maintain context from previous conversations and can refer back to earlier topics discussed.
        When users ask for more details or clarification (like "give in detail", "explain more", "elaborate"), 
        refer to the previous conversation context to understand what they're asking about.
        Format your responses in markdown for readability. 
        Include code examples when relevant. 
        Be direct and to the point while remaining helpful."""
        
        # Format conversation history for the model
        formatted_history = format_conversation_history(chat_history)
        
        # Start chat with history if available
        if formatted_history:
            chat = model.start_chat(history=formatted_history)
            response = chat.send_message(prompt)
        else:
            # No history, start fresh chat with system prompt
            chat = model.start_chat(history=[])
            response = chat.send_message(f"{system_prompt}\n\nUser question: {prompt}")
        
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
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
            # Get current chat history for context
            chat_history = get_chat_history()
            
            # Check for duplicate question
            if chat_history and chat_history[-1].get('question') == user_input:
                flash("This question was already answered. See below.", "warning")
                return redirect(url_for('home'))

            # Generate response with conversation context
            start_time = time.time()
            response_text = generate_ai_response(user_input, chat_history)
            response_time = time.time() - start_time
            
            # Convert markdown to HTML
            response_html = Markup(markdown.markdown(
                response_text,
                extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
            ))

            # Add to chat history with both raw text and HTML
            chat_history.append({
                'question': user_input,
                'response': response_html,  # HTML for display
                'response_text': response_text,  # Raw text for AI context
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

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for AJAX chat requests with conversation context"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'success': False, 'error': 'No message provided'}), 400
    
    user_input = sanitize_input(data['message'])
    
    if not model:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 500
    
    try:
        # Get current chat history for context
        chat_history = get_chat_history()
        
        # Generate response with conversation context
        start_time = time.time()
        response_text = generate_ai_response(user_input, chat_history)
        response_time = time.time() - start_time
        
        response_html = markdown.markdown(
            response_text,
            extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
        )
        
        # Add to chat history with both raw text and HTML
        chat_history.append({
            'question': user_input,
            'response': Markup(response_html),  # HTML for display
            'response_text': response_text,  # Raw text for AI context
            'response_time': f"{response_time:.2f}s",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_chat_history(chat_history[-50:])
        
        return jsonify({
            'success': True,
            'response': response_text,
            'response_html': response_html,
            'response_time': f"{response_time:.2f}s",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_initialized': model is not None,
        'api_key_configured': 'GOOGLE_API_KEY' in os.environ,
        'cache_size': len(response_cache),
        'chat_history_count': len(get_chat_history()),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

