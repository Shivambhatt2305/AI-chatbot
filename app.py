from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import os
import google.generativeai as genai
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from functools import wraps
from datetime import timedelta
from urllib.parse import urlparse

# Get database URL from Render environment
DATABASE_URL = os.environ.get('DATABASE_URL')


# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=1)  # Session expires after 1 day

# Configure Gemini API
API_KEY = os.environ.get('GEMINI_API_KEY', "AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I")
os.environ['GOOGLE_API_KEY'] = API_KEY
genai.configure(api_key=API_KEY)

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'database': os.environ.get('DB_NAME', 'chatbot_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', '122405'),
    'port': os.environ.get('DB_PORT', '1369')
}


# Initialize model
model = None
try:
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction="""You are a helpful AI assistant. You maintain context from previous messages in the conversation. 
        When users ask follow-up questions like "explain in detail", "give me more info", "elaborate", etc., 
        refer back to the previous topics discussed in the conversation."""
    )
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model = None

# Database connection function with retry logic
def get_db_connection(max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            print("Database connection established")
            return conn
        except psycopg2.OperationalError as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries reached, giving up")
                return None
        except Exception as e:
            print(f"Unexpected database error: {str(e)}")
            return None

# Test database connection
def test_db_connection():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT version();')
                version = cur.fetchone()
                print(f"Database connected successfully: {version['version']}")
                return True
        except Exception as e:
            print(f"Database test error: {str(e)}")
            return False
        finally:
            conn.close()
    return False

# Initialize database tables
def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Create users table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        email VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create chat_sessions table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create chat_messages table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cur.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON chat_messages(user_id)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)')
                
                conn.commit()
                print("Database tables initialized successfully")
                return True
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    return False

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/', methods=['GET'])
@login_required
def home():
    user_id = session.get('user_id')
    username = session.get('username')
    
    # Get user's chat history
    conn = get_db_connection()
    chat_history = []
    
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT message, response, to_char(timestamp, 'HH24:MI:SS') as timestamp
                    FROM chat_messages
                    WHERE user_id = %s
                    ORDER BY timestamp ASC
                ''', (user_id,))
                
                results = cur.fetchall()
                for row in results:
                    chat_history.append({
                        'question': row['message'],
                        'response': row['response'],
                        'timestamp': row['timestamp']
                    })
        except Exception as e:
            print(f"Error fetching chat history: {str(e)}")
            flash('Error loading chat history', 'error')
        finally:
            conn.close()
    
    return render_template('index.html', chat_history=chat_history, username=username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                flash('Database connection error. Please try again later.', 'error')
                return render_template('login.html')
            
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM users WHERE username = %s', (username,))
                user = cur.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session.permanent = True
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    
                    next_page = request.args.get('next')
                    if next_page:
                        return redirect(next_page)
                    return redirect(url_for('home'))
                else:
                    flash('Invalid username or password', 'error')
        except psycopg2.Error as e:
            flash(f'Database error during login: {str(e)}', 'error')
            print(f"Login error: {str(e)}")
        except Exception as e:
            flash('An unexpected error occurred', 'error')
            print(f"Unexpected error: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        return render_template('login.html')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        email = request.form.get('email', '').strip()
        
        # Validation
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Database operations
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                flash('Database connection error. Please try again later.', 'error')
                return render_template('register.html')
            
            with conn.cursor() as cur:
                # Check if username exists
                cur.execute('SELECT id FROM users WHERE username = %s', (username,))
                if cur.fetchone():
                    flash('Username already exists', 'error')
                    return render_template('register.html')
                
                # Check if email exists (if provided)
                if email:
                    cur.execute('SELECT id FROM users WHERE email = %s', (email,))
                    if cur.fetchone():
                        flash('Email already exists', 'error')
                        return render_template('register.html')
                
                # Insert new user
                cur.execute(
                    'INSERT INTO users (username, password, email) VALUES (%s, %s, %s) RETURNING id',
                    (username, hashed_password, email if email else None)
                )
                user_id = cur.fetchone()['id']
                conn.commit()
                
                # Log user in
                session.permanent = True
                session['user_id'] = user_id
                session['username'] = username
                
                flash('Registration successful! Welcome to Gemini Chat!', 'success')
                return redirect(url_for('home'))
                
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            flash(f'Database error during registration: {str(e)}', 'error')
            print(f"Registration error: {str(e)}")
        except Exception as e:
            flash('An unexpected error occurred', 'error')
            print(f"Unexpected error: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    user_id = session.get('user_id')
    data = request.get_json()
    user_input = data.get('message', '').strip()
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    if user_input.lower() == 'exit':
        return jsonify({"response": "Chat session ended."})
    
    if not model:
        return jsonify({"error": "AI model not initialized. Please check API key and model availability."})
    
    try:
        # Get user's chat history for context
        conn = get_db_connection()
        recent_history = []
        session_id = None
        
        if conn:
            try:
                with conn.cursor() as cur:
                    # Get or create a chat session for this user
                    cur.execute('''
                        SELECT id FROM chat_sessions 
                        WHERE user_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    ''', (user_id,))
                    
                    session_row = cur.fetchone()
                    if session_row:
                        session_id = session_row['id']
                    else:
                        cur.execute(
                            'INSERT INTO chat_sessions (user_id) VALUES (%s) RETURNING id',
                            (user_id,)
                        )
                        session_id = cur.fetchone()['id']
                        conn.commit()
                    
                    # Get recent messages for context (last 5 messages)
                    cur.execute('''
                        SELECT message, response
                        FROM chat_messages
                        WHERE user_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 5
                    ''', (user_id,))
                    
                    for row in cur.fetchall():
                        recent_history.append({
                            'user': row['message'],
                            'ai': row['response']
                        })
                    
                    # Reverse to get chronological order
                    recent_history.reverse()
            except Exception as e:
                print(f"Error retrieving chat history: {str(e)}")
        
        # Create chat session with context
        chat_session = model.start_chat(history=[])
        
        # Add context from recent history if available
        if recent_history:
            context_prompt = "Previous conversation context (for reference only, don't repeat):\n"
            for item in recent_history:
                context_prompt += f"User: {item['user']}\nAI: {item['ai']}\n\n"
            context_prompt += f"Current question: {user_input}"
            
            # Send the message with context
            response = chat_session.send_message(context_prompt)
        else:
            # Send message without context for new users
            response = chat_session.send_message(user_input)
        
        response_text = response.text
        
        # Save message and response to database
        if conn and session_id:
            try:
                with conn.cursor() as cur:
                    cur.execute('''
                        INSERT INTO chat_messages (session_id, user_id, message, response)
                        VALUES (%s, %s, %s, %s)
                    ''', (session_id, user_id, user_input, response_text))
                    conn.commit()
            except Exception as e:
                print(f"Error saving message: {str(e)}")
            finally:
                conn.close()
        
        return jsonify({
            "response": response_text,
            "timestamp": time.strftime('%H:%M:%S')
        })
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"Detailed error: {e}")
        return jsonify({"error": error_message})

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    user_id = session.get('user_id')
    
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Delete all messages for this user
                cur.execute('DELETE FROM chat_messages WHERE user_id = %s', (user_id,))
                # Delete all sessions for this user
                cur.execute('DELETE FROM chat_sessions WHERE user_id = %s', (user_id,))
                conn.commit()
                print(f"Cleared history for user {user_id}")
        except Exception as e:
            print(f"Error clearing history: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    return jsonify({"status": "success", "message": "Chat history cleared"})

@app.route('/health')
def health_check():
    """Health check endpoint to verify system status"""
    db_status = test_db_connection()
    model_status = model is not None
    
    return jsonify({
        "status": "healthy" if db_status and model_status else "unhealthy",
        "database": "connected" if db_status else "disconnected",
        "ai_model": "initialized" if model_status else "not initialized",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })

# Initialize database and test connections on startup
if __name__ == '__main__':
    print("Starting Gemini Chat Application...")
    
    # Test database connection
    if test_db_connection():
        print("✓ Database connection successful")
        
        # Initialize database tables
        if init_db():
            print("✓ Database tables initialized")
        else:
            print("✗ Failed to initialize database tables")
    else:
        print("✗ Database connection failed")
        print("Please check your database configuration in DB_CONFIG")
    
    # Test AI model
    if model:
        print("✓ Gemini AI model initialized")
    else:
        print("✗ Failed to initialize Gemini AI model")
        print("Please check your API key")
    
    print("\nStarting Flask server...")
    print("Visit http://localhost:5000 to access the application")
    print("Visit http://localhost:5000/health to check system status")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
