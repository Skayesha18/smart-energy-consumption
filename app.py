from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import sqlite3
from functools import wraps

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'smart-energy-ai-secret-key-2024'

# Updated CORS to allow your Render URL
CORS(app, origins=[
    'http://localhost:5000', 
    'http://127.0.0.1:5000', 
    'http://localhost:3000',
    'https://smart-energy-consumption-2.onrender.com'  # Your Render URL
])

# =============================================
# EMAIL CONFIGURATION - UPDATE THESE!
# =============================================
SENDER_EMAIL = "teamsmartenergy12@gmail.com"  # Your email
SENDER_PASSWORD = "yzut aang kjfm sxud"      # Your Gmail app password
# =============================================

# Load model (if exists)
model = None
try:
    if os.path.exists('smart_energy.pkl'):
        with open('smart_energy.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model file not found, using simulation mode")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Database setup
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('smart_energy.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        joined_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin INTEGER DEFAULT 0
    )''')
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        date TEXT,
        time TEXT,
        temperature REAL,
        occupancy INTEGER,
        humidity REAL,
        area REAL,
        devices TEXT,
        prediction REAL,
        cost REAL,
        carbon REAL,
        savings_potential REAL,
        peak_hours TEXT,
        source TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Contacts table
    c.execute('''CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending'
    )''')
    
    # Reviews table
    c.execute('''CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        rating INTEGER,
        comment TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Queries table (for AI assistant)
    c.execute('''CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        email TEXT,
        query TEXT,
        response TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Gamification data table
    c.execute('''CREATE TABLE IF NOT EXISTS gamification (
        user_id INTEGER PRIMARY KEY,
        energy_score INTEGER DEFAULT 0,
        level TEXT DEFAULT 'bronze',
        badges TEXT DEFAULT '["beginner"]',
        current_challenge TEXT,
        total_energy_saved REAL DEFAULT 0,
        total_money_saved REAL DEFAULT 0,
        total_co2_saved REAL DEFAULT 0,
        streak_days INTEGER DEFAULT 0,
        last_activity DATE,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize database
init_db()

# Feature names for the model
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'Occupancy', 'HVACUsage', 
    'LightingUsage', 'RenewableEnergy', 'DayOfWeek', 'Holiday',
    'Zscore', 'IsWeekend', 'Total_Device_Power', 'High_Load',
    'Rolling_Mean_3', 'Rolling_Mean_24'
]

# Device list for power calculation
DEVICES = [
    ("Air Conditioner", 1500),
    ("Refrigerator", 150),
    ("LED TV", 100),
    ("Washing Machine", 500),
    ("Microwave", 1000),
    ("Laptop", 50),
    ("Desktop PC", 200),
    ("Lighting", 100),
    ("Electric Heater", 2000),
    ("Electric Oven", 2000),
    ("Dishwasher", 1200),
    ("Water Heater", 3000),
    ("Ceiling Fan", 75),
    ("Charger", 25),
    ("Gaming Console", 150)
]

# Helper functions
def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('smart_energy.db')
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Please login first'}), 401
        return f(*args, **kwargs)
    return decorated_function

def engineer_features(data):
    """Feature engineering for prediction"""
    try:
        features = {
            'Temperature': float(data.get('temperature', 25)),
            'Humidity': float(data.get('humidity', 50)),
            'Occupancy': int(data.get('occupancy', 2)),
            'HVACUsage': 1 if str(data.get('hvac', 'off')).lower() == 'on' else 0,
            'LightingUsage': 1 if str(data.get('lighting', 'off')).lower() == 'on' else 0,
            'RenewableEnergy': float(data.get('renewable', 10)),
            'DayOfWeek': int(data.get('day', 0)),
            'Holiday': 0,
            'Zscore': 0,
            'IsWeekend': 1 if int(data.get('day', 0)) in [5, 6] else 0,
            'Total_Device_Power': float(data.get('renewable', 10)) * 0.8,
            'High_Load': 1 if float(data.get('temperature', 25)) > 25 and int(data.get('occupancy', 2)) > 3 else 0,
            'Rolling_Mean_3': 0,
            'Rolling_Mean_24': 0
        }
        
        feature_array = [features[name] for name in FEATURE_NAMES]
        return np.array(feature_array).reshape(1, -1)
        
    except Exception as e:
        print(f"Feature engineering error: {e}")
        return None

def calculate_prediction(data):
    """Calculate energy consumption prediction"""
    try:
        temperature = float(data.get('temperature', 25))
        occupancy = int(data.get('occupancy', 2))
        area = float(data.get('area', 1200))
        devices = data.get('devices', [])
        hour = int(data.get('hour', 14))
        
        # Calculate device power
        device_power = 0
        for device_name in devices:
            for name, power in DEVICES:
                if isinstance(device_name, str) and device_name.lower() in name.lower():
                    device_power += power
                    break
        
        # Base calculation
        base_consumption = (area / 100) * 0.5
        occupancy_factor = occupancy * 0.2
        temp_factor = abs(22 - temperature) * 0.15
        device_factor = (device_power / 1000) * 1.2
        
        # Time factor
        if 6 <= hour <= 9:
            time_factor = 1.1
        elif 18 <= hour <= 21:
            time_factor = 1.4
        elif hour >= 22 or hour <= 5:
            time_factor = 0.8
        else:
            time_factor = 1.0
        
        prediction = (base_consumption + occupancy_factor + temp_factor + device_factor) * time_factor
        prediction = max(0.5, prediction)
        
        return prediction, device_power
        
    except Exception as e:
        print(f"Prediction calculation error: {e}")
        return 45.5, 0

def send_email(to_email, subject, body):
    """Send email with error handling"""
    try:
        if not SENDER_EMAIL or SENDER_EMAIL == "your_email@gmail.com":
            print("❌ Email not configured properly")
            return False, "Email service not configured"
        
        if not SENDER_PASSWORD:
            print("❌ Email password not set")
            return False, "Email password not set"
        
        print(f"📧 Attempting to send email to: {to_email}")
        print(f"   From: {SENDER_EMAIL}")
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ Email sent successfully via SSL to {to_email}")
            return True, "Email sent successfully"
            
        except Exception as e1:
            print(f"   SSL failed: {str(e1)[:100]}")
            
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                server.quit()
                print(f"✅ Email sent successfully via TLS to {to_email}")
                return True, "Email sent successfully"
                
            except Exception as e2:
                print(f"   TLS failed: {str(e2)}")
                return False, f"Failed to send email: {str(e2)}"
                
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False, f"Email error: {str(e)}"

# ===================== AUTHENTICATION ROUTES =====================

@app.route('/api/register', methods=['POST'])
def api_register():
    """Register a new user"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        
        # Validation
        if not all([username, password, name, email]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        
        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        conn = get_db_connection()
        
        # Check if username exists
        user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if user:
            conn.close()
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        
        # Check if email exists
        user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if user:
            conn.close()
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Create user
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
            (username, hashed_password, name, email)
        )
        conn.commit()
        
        # Get new user ID
        user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        
        # Initialize gamification data
        conn.execute(
            '''INSERT INTO gamification (user_id, energy_score, level, badges, streak_days, last_activity)
               VALUES (?, 0, 'bronze', '["beginner"]', 1, date('now'))''',
            (user['id'],)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Registration successful! Please login.'})
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT id, username, password, name, email, joined_date, is_admin FROM users WHERE username = ?',
            (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['name'] = user['name']
            session['email'] = user['email']
            session['is_admin'] = user['is_admin']
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'name': user['name'],
                    'email': user['email'],
                    'joinedDate': user['joined_date'],
                    'is_admin': user['is_admin']
                }
            })
        
        return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is logged in"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session.get('username'),
                'name': session.get('name'),
                'email': session.get('email'),
                'is_admin': session.get('is_admin', False)
            }
        })
    return jsonify({'authenticated': False})

# ===================== CONTACT ROUTES =====================

@app.route('/api/contact', methods=['POST'])
def api_contact():
    """API endpoint for contact form"""
    try:
        data = request.json
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()
        
        print(f"📝 Contact form submitted:")
        print(f"   Name: {name}")
        print(f"   Email: {email}")
        print(f"   Message: {message[:50]}...")
        
        # Validation
        if not name or not email or not message:
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        # Store in database
        conn = get_db_connection()
        cursor = conn.execute(
            'INSERT INTO contacts (name, email, message, status) VALUES (?, ?, ?, ?)',
            (name, email, message, 'pending')
        )
        contact_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Prepare email body
        email_body = f"""
Dear {name},

Thank you for contacting Smart Energy Prediction Platform!

We have received your message:
"{message}"

Contact ID: #{contact_id}
Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

Our team will review your message and get back to you within 24-48 hours.

Best regards,
Smart Energy Prediction Platform Team
Email: {SENDER_EMAIL}
        """
        
        # Send confirmation email
        email_sent, email_message = send_email(
            email,
            f"Thank you for contacting us (Ref: #{contact_id})",
            email_body
        )
        
        if email_sent:
            return jsonify({
                'success': True,
                'message': 'Thank you! Your message has been sent. Check your email for confirmation.',
                'contact_id': contact_id
            })
        else:
            return jsonify({
                'success': True,
                'message': f'Message received! (Email notification failed: {email_message})',
                'contact_id': contact_id
            })
            
    except Exception as e:
        print(f"❌ Contact API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===================== PREDICTION ROUTES =====================

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for energy prediction"""
    try:
        data = request.json
        
        # Get parameters
        temperature = float(data.get('temperature', 25))
        occupancy = int(data.get('occupancy', 2))
        humidity = float(data.get('humidity', 50))
        area = float(data.get('area', 1200))
        devices = data.get('devices', [])
        date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        time_str = data.get('time', '14:00')
        
        # Calculate hour
        hour = 14
        try:
            hour = int(time_str.split(':')[0])
        except:
            pass
        
        # Calculate prediction
        if model:
            features = {
                'temperature': temperature,
                'humidity': humidity,
                'occupancy': occupancy,
                'hvac': 'on' if 'Air Conditioner' in devices else 'off',
                'lighting': 'on' if 'Lighting' in devices else 'off',
                'renewable': 10,
                'day': datetime.now().weekday(),
                'hour': hour
            }
            feature_array = engineer_features(features)
            if feature_array is not None:
                prediction = model.predict(feature_array)[0]
            else:
                prediction, _ = calculate_prediction(features)
        else:
            features = {
                'temperature': temperature,
                'occupancy': occupancy,
                'area': area,
                'devices': devices,
                'hour': hour
            }
            prediction, device_power = calculate_prediction(features)
        
        # Calculate additional metrics
        cost_per_kwh = 0.15 * 83  # INR conversion
        estimated_cost = prediction * cost_per_kwh
        carbon_footprint = prediction * 0.5
        
        # Calculate savings potential
        device_power = 0
        for device_name in devices:
            for name, power in DEVICES:
                if device_name.lower() in name.lower():
                    device_power += power
                    break
        
        if device_power:
            savings_potential = min(30, (device_power / 5000) * 30)
        else:
            savings_potential = 15
        
        # Determine peak hours
        if 18 <= hour <= 21:
            peak_hours = '6-9 PM (Peak Hours)'
        elif 6 <= hour <= 9:
            peak_hours = '6-9 AM (Morning Peak)'
        else:
            peak_hours = 'Normal Hours'
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.execute(
            '''INSERT INTO predictions 
               (user_id, date, time, temperature, occupancy, humidity, area, devices, 
                prediction, cost, carbon, savings_potential, peak_hours, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (session['user_id'], date_str, time_str, temperature, occupancy, humidity, area,
             json.dumps(devices), prediction, estimated_cost, carbon_footprint,
             savings_potential, peak_hours, 'backend_api')
        )
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update user statistics
        update_user_stats(session['user_id'])
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'prediction': float(prediction),
            'cost': float(estimated_cost),
            'carbon': float(carbon_footprint),
            'savings_potential': float(savings_potential),
            'peak_hours': peak_hours,
            'unit': 'kWh',
            'currency': '₹'
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Prediction failed: {str(e)}"}), 400

@app.route('/api/predictions', methods=['GET'])
@login_required
def get_predictions_api():
    """Get user's recent predictions"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        conn = get_db_connection()
        predictions = conn.execute(
            '''SELECT id, timestamp, date, time, temperature, occupancy, area, devices,
                      prediction, cost, carbon, savings_potential, peak_hours, source
               FROM predictions 
               WHERE user_id = ? 
               ORDER BY timestamp DESC 
               LIMIT ?''',
            (session['user_id'], limit)
        ).fetchall()
        conn.close()
        
        result = []
        for pred in predictions:
            result.append({
                'id': pred['id'],
                'timestamp': pred['timestamp'],
                'date': pred['date'],
                'time': pred['time'],
                'temperature': pred['temperature'],
                'occupancy': pred['occupancy'],
                'area': pred['area'],
                'devices': json.loads(pred['devices']) if pred['devices'] else [],
                'prediction': pred['prediction'],
                'cost': pred['cost'],
                'carbon': pred['carbon'],
                'savings_potential': pred['savings_potential'],
                'peak_hours': pred['peak_hours'],
                'source': pred['source']
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify([]), 500

def update_user_stats(user_id):
    """Update user statistics"""
    try:
        conn = get_db_connection()
        
        # Get total predictions count
        count = conn.execute(
            'SELECT COUNT(*) as count FROM predictions WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        
        # Get total energy saved (based on savings potential)
        predictions = conn.execute(
            'SELECT prediction, cost, savings_potential FROM predictions WHERE user_id = ?',
            (user_id,)
        ).fetchall()
        
        total_energy = sum(p['prediction'] for p in predictions)
        total_cost = sum(p['cost'] for p in predictions)
        
        conn.close()
        
        # Update gamification based on total energy
        update_gamification(user_id, total_energy, total_cost)
        
    except Exception as e:
        print(f"Error updating user stats: {e}")

# ===================== GAMIFICATION ROUTES =====================

def update_gamification(user_id, total_energy, total_cost):
    """Update gamification data based on user activity"""
    try:
        conn = get_db_connection()
        
        # Get current gamification data
        gam = conn.execute(
            'SELECT energy_score, level, badges FROM gamification WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        
        if not gam:
            # Initialize if not exists
            conn.execute(
                '''INSERT INTO gamification (user_id, energy_score, level, badges, 
                    total_energy_saved, total_money_saved, streak_days, last_activity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, date('now'))''',
                (user_id, 0, 'bronze', '["beginner"]', 0, 0, 1)
            )
            conn.commit()
            conn.close()
            return
        
        # Calculate points based on energy saved (assuming 15% savings potential)
        energy_saved = total_energy * 0.15
        money_saved = total_cost * 0.15
        
        # Update points (1 point per 5 kWh saved)
        points_earned = int(energy_saved / 5)
        
        new_score = gam['energy_score'] + points_earned
        
        # Determine level
        if new_score >= 800:
            level = 'gold'
        elif new_score >= 500:
            level = 'silver'
        elif new_score >= 200:
            level = 'bronze'
        else:
            level = gam['level']
        
        # Update badges
        badges = json.loads(gam['badges']) if gam['badges'] else ['beginner']
        
        if level == 'bronze' and 'bronze' not in badges:
            badges.append('bronze')
        if level == 'silver' and 'silver' not in badges:
            badges.append('silver')
        if level == 'gold' and 'gold' not in badges:
            badges.append('gold')
        
        # Update database
        conn.execute(
            '''UPDATE gamification 
               SET energy_score = ?, level = ?, badges = ?,
                   total_energy_saved = total_energy_saved + ?,
                   total_money_saved = total_money_saved + ?,
                   streak_days = streak_days + 1,
                   last_activity = date('now')
               WHERE user_id = ?''',
            (new_score, level, json.dumps(badges), energy_saved, money_saved, user_id)
        )
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating gamification: {e}")

@app.route('/api/gamification', methods=['GET'])
@login_required
def get_gamification():
    """Get user's gamification data"""
    try:
        conn = get_db_connection()
        
        gam = conn.execute(
            '''SELECT energy_score, level, badges, total_energy_saved, total_money_saved, 
                      total_co2_saved, streak_days, last_activity
               FROM gamification WHERE user_id = ?''',
            (session['user_id'],)
        ).fetchone()
        
        if not gam:
            # Initialize
            conn.execute(
                '''INSERT INTO gamification (user_id, energy_score, level, badges, streak_days, last_activity)
                   VALUES (?, 0, 'bronze', '["beginner"]', 1, date('now'))''',
                (session['user_id'],)
            )
            conn.commit()
            gam = conn.execute(
                'SELECT * FROM gamification WHERE user_id = ?',
                (session['user_id'],)
            ).fetchone()
        
        # Get current challenge
        challenge = get_current_challenge(session['user_id'])
        
        conn.close()
        
        return jsonify({
            'success': True,
            'energy_score': gam['energy_score'],
            'level': gam['level'],
            'badges': json.loads(gam['badges']) if gam['badges'] else ['beginner'],
            'stats': {
                'totalEnergySaved': gam['total_energy_saved'],
                'totalMoneySaved': gam['total_money_saved'],
                'totalCO2Saved': gam['total_co2_saved'],
                'streakDays': gam['streak_days']
            },
            'current_challenge': challenge
        })
        
    except Exception as e:
        print(f"Error getting gamification: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_current_challenge(user_id):
    """Get current challenge for user"""
    # For now, return a default challenge
    # In production, this would be stored in the database
    return {
        'type': 'peak_hour_reduction',
        'name': 'Reduce Peak Hour Usage',
        'description': 'Use 20% less energy during peak hours (6-9 PM) this week',
        'target': 20,
        'progress': 65,
        'reward': 100
    }

@app.route('/api/gamification/update', methods=['POST'])
@login_required
def update_gamification_score():
    """Update gamification score based on action"""
    try:
        data = request.json
        action = data.get('action', '')
        points = data.get('points', 0)
        
        if action == 'prediction':
            points = 10
        elif action == 'review':
            points = 5
        elif action == 'contact':
            points = 5
        
        conn = get_db_connection()
        conn.execute(
            'UPDATE gamification SET energy_score = energy_score + ? WHERE user_id = ?',
            (points, session['user_id'])
        )
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'points_earned': points})
        
    except Exception as e:
        print(f"Error updating gamification score: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===================== REVIEW ROUTES =====================

@app.route('/api/review', methods=['POST'])
@login_required
def submit_review_api():
    """Submit a review"""
    try:
        data = request.json
        rating = int(data.get('rating', 5))
        comment = data.get('comment', '')
        
        if rating < 1 or rating > 5:
            return jsonify({'success': False, 'error': 'Rating must be between 1 and 5'}), 400
        
        conn = get_db_connection()
        conn.execute(
            '''INSERT INTO reviews (user_id, name, rating, comment) 
               VALUES (?, ?, ?, ?)''',
            (session['user_id'], session['name'], rating, comment)
        )
        conn.commit()
        conn.close()
        
        # Award points for review
        update_gamification_score({'action': 'review'})
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your review!'
        })
        
    except Exception as e:
        print(f"Error submitting review: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/reviews', methods=['GET'])
def get_reviews_api():
    """Get recent reviews"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        conn = get_db_connection()
        reviews = conn.execute(
            '''SELECT id, name, rating, comment, timestamp 
               FROM reviews 
               ORDER BY timestamp DESC 
               LIMIT ?''',
            (limit,)
        ).fetchall()
        conn.close()
        
        result = []
        for review in reviews:
            result.append({
                'id': review['id'],
                'name': review['name'],
                'rating': review['rating'],
                'comment': review['comment'],
                'timestamp': review['timestamp']
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting reviews: {e}")
        return jsonify([]), 500

# ===================== QUERY ROUTES (AI Assistant) =====================

@app.route('/api/query', methods=['POST'])
@login_required
def submit_query_api():
    """Submit a query to AI assistant"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400
        
        # Generate AI response based on query
        response = generate_ai_response(query_text)
        
        # Store in database
        conn = get_db_connection()
        cursor = conn.execute(
            '''INSERT INTO queries (user_id, email, query, response, status) 
               VALUES (?, ?, ?, ?, ?)''',
            (session['user_id'], session['email'], query_text, response, 'resolved')
        )
        query_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'query_id': query_id,
            'response': response
        })
        
    except Exception as e:
        print(f"Error submitting query: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """Chatbot endpoint - no login required"""
    try:
        data = request.json
        user_message = data.get('message', '').lower().strip()
        
        response = generate_ai_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

def generate_ai_response(query):
    """Generate AI response for user queries"""
    query_lower = query.lower()
    
    # Prediction related
    if any(word in query_lower for word in ['predict', 'prediction', 'consumption', 'usage']):
        return """To make an energy prediction:
1. Go to the **Predict** section from the navigation menu
2. Enter your details:
   - Temperature
   - Number of occupants
   - Area in sq ft
   - Date and time
   - Select active devices
3. Click "Predict Energy Consumption"
4. View your prediction, estimated cost, and carbon footprint

You can also upload a data file (.txt, .csv, .json) for automatic prediction!"""
    
    # Energy saving tips
    elif any(word in query_lower for word in ['save', 'reduce', 'bill', 'efficient']):
        return """**Top Energy Saving Tips:** 💡

**Immediate Actions:**
• Set thermostat to 24°C in summer, 20°C in winter
• Use LED bulbs (saves 75% energy)
• Unplug devices when not in use
• Run washing machine/dishwasher at night
• Use natural light during daytime

**Medium-term Actions:**
• Install programmable thermostat
• Seal windows and doors
• Clean AC filters monthly
• Use power strips for electronics

**Long-term Investments:**
• Consider solar panels
• Upgrade to Energy Star appliances
• Improve home insulation
• Install smart home systems

Potential savings: 15-30% on energy bills!"""
    
    # Temperature related
    elif any(word in query_lower for word in ['temperature', 'thermostat', 'ac', 'cooling', 'heating']):
        return """**Temperature Optimization:** 🌡️

**Optimal Settings:**
• Summer: 24-26°C (Saves 6-8% per degree higher)
• Winter: 18-20°C (Saves 3-5% per degree lower)
• Night: 22°C (summer) / 16°C (winter)

**Smart Tips:**
• Each 1°C adjustment saves 3-5% on HVAC energy
• Use ceiling fans to feel 4°C cooler
• Close blinds during hottest part of day
• Use zone heating/cooling if available
• Install ceiling fans for better air circulation

**Seasonal Tips:**
• Spring/Fall: Use natural ventilation
• Winter: Open curtains during sunny days
• Summer: Use window reflectors"""
    
    # Peak hours
    elif any(word in query_lower for word in ['peak', 'hour', 'time']):
        return """**Understanding Peak Hours:** ⏰

**Peak Hours:** 6 PM - 9 PM
• Electricity rates are 30-50% higher
• Grid demand is at maximum
• Environmental impact is greater

**Off-Peak Hours:** 10 PM - 6 AM
• Lower electricity rates
• Reduced grid stress
• Better for the environment

**Smart Strategies:**
✅ Run appliances during off-peak hours
✅ Pre-cool your home before peak hours
✅ Use delay start features
✅ Consider battery storage for peak shaving
✅ Charge EVs overnight

**Potential Savings:** 15-30% by shifting usage!"""
    
    # Devices
    elif any(word in query_lower for word in ['device', 'appliance', 'equipment']):
        return """**Device Energy Usage Guide:** 📱

**High-Power Devices (Use wisely):**
• Air Conditioner: 1500-2000W
• Water Heater: 3000W
• Electric Oven: 2000W
• Washing Machine: 500W
• Dishwasher: 1200W

**Medium-Power Devices:**
• Refrigerator: 150W
• Microwave: 1000W
• Desktop PC: 200W
• TV: 100W

**Low-Power Devices:**
• LED Lighting: 10-20W
• Laptop: 50W
• Chargers: 5-25W
• Ceiling Fan: 75W

**Tips:**
• Unplug devices when not in use
• Use power strips for electronics
• Replace old appliances with Energy Star models
• Consider smart plugs for monitoring"""
    
    # Carbon footprint
    elif any(word in query_lower for word in ['carbon', 'footprint', 'co2', 'environment']):
        return """**Reducing Your Carbon Footprint:** 🌍

**Current Impact:**
• 1 kWh = 0.5 kg CO₂
• Average home: 900 kg CO₂/month
• Equivalent to 45 trees needed

**Ways to Reduce:**
1. **Energy Efficiency**
   • LED lighting (75% less energy)
   • Energy Star appliances
   • Proper insulation

2. **Renewable Energy**
   • Solar panels
   • Green power programs
   • Community solar

3. **Behavior Changes**
   • Reduce peak hour usage
   • Unplug devices
   • Air dry clothes
   • Lower water heater temperature

**Impact of 20% Reduction:**
• 180 kg CO₂ saved monthly
• Equivalent to planting 9 trees
• Saves ₹300-500 monthly"""
    
    # Project information
    elif any(word in query_lower for word in ['project', 'about', 'smartenergy']):
        return """**SmartEnergy AI - Project Overview** 🚀

**What is SmartEnergy AI?**
An advanced energy management platform using AI to predict, analyze, and optimize energy consumption for homes and businesses.

**Key Features:**
✅ **AI-Powered Predictions** - Accurate consumption forecasting
✅ **Real-time Analytics** - Interactive dashboards with charts
✅ **Cost Optimization** - Recommendations to reduce bills
✅ **Carbon Tracking** - Monitor environmental impact
✅ **Voice-enabled Assistant** - Hands-free interaction
✅ **Gamification** - Earn points and badges for saving energy

**Technology Stack:**
• Frontend: HTML5, CSS3, JavaScript
• Backend: Flask (Python)
• Database: SQLite
• Charts: Chart.js
• AI: Machine Learning models

**Benefits:**
💰 Save up to 30% on energy bills
🌱 Reduce carbon footprint
📊 Data-driven insights
⏱️ Quick and accurate predictions

**"Empowering a sustainable future through intelligent energy management."**"""
    
    # General help
    elif any(word in query_lower for word in ['help', 'how to', 'guide']):
        return """**How to Use SmartEnergy AI:** 📚

**Quick Start Guide:**

1. **Make a Prediction:**
   • Click "Predict" in navigation
   • Enter your parameters
   • Click "Predict Energy Consumption"
   • View results and recommendations

2. **Track Progress:**
   • View Dashboard for analytics
   • Check prediction history
   • Monitor energy usage trends

3. **Get AI Assistance:**
   • Ask me questions
   • Use voice input (microphone button)
   • Get step-by-step guidance

4. **Earn Rewards:**
   • Complete challenges
   • Earn points and badges
   • Track your energy-saving progress

5. **Contact Support:**
   • Use Contact Us form
   • We'll respond within 24 hours
   • Share feedback and suggestions

**Need more help?** Just ask me anything about energy management!"""
    
    # Default response
    else:
        return """**How can I help you?** 🤖

I can assist you with:

🔮 **Energy Predictions**
• "How to make a prediction?"
• "Predict energy usage for me"

💡 **Saving Tips**
• "How to reduce energy bill?"
• "Energy saving tips"
• "Best temperature for AC"

⏰ **Peak Hours**
• "What are peak hours?"
• "When is electricity cheaper?"

📱 **Devices**
• "Which devices use most power?"
• "How to save with appliances"

🌱 **Environment**
• "How to reduce carbon footprint?"
• "What is my environmental impact?"

📊 **Project**
• "Tell me about this project"
• "How does this work?"

Just type your question and I'll help you! 🎯"""
    
    return response

# ===================== DASHBOARD ROUTES =====================

@app.route('/api/dashboard/stats', methods=['GET'])
@login_required
def get_dashboard_stats():
    """Get dashboard statistics for user"""
    try:
        conn = get_db_connection()
        
        # Get predictions
        predictions = conn.execute(
            '''SELECT prediction, cost, carbon, timestamp, date
               FROM predictions 
               WHERE user_id = ? 
               ORDER BY timestamp DESC 
               LIMIT 30''',
            (session['user_id'],)
        ).fetchall()
        
        # Calculate totals
        total_energy = sum(p['prediction'] for p in predictions)
        total_cost = sum(p['cost'] for p in predictions)
        total_carbon = sum(p['carbon'] for p in predictions)
        
        # Get weekly trend
        from datetime import timedelta
        week_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
        
        weekly_predictions = [p for p in predictions if p['timestamp'] and datetime.fromisoformat(p['timestamp']) > week_ago]
        
        weekly_energy = sum(p['prediction'] for p in weekly_predictions)
        weekly_cost = sum(p['cost'] for p in weekly_predictions)
        
        # Get gamification data
        gam = conn.execute(
            'SELECT energy_score, level FROM gamification WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_energy': total_energy,
                'total_cost': total_cost,
                'total_carbon': total_carbon,
                'weekly_energy': weekly_energy,
                'weekly_cost': weekly_cost,
                'total_predictions': len(predictions),
                'energy_score': gam['energy_score'] if gam else 0,
                'level': gam['level'] if gam else 'bronze'
            }
        })
        
    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===================== HEALTH CHECK =====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    conn = get_db_connection()
    user_count = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()
    prediction_count = conn.execute('SELECT COUNT(*) as count FROM predictions').fetchone()
    conn.close()
    
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Energy Prediction Platform',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'database': 'connected',
        'stats': {
            'users': user_count['count'],
            'predictions': prediction_count['count']
        },
        'email_configured': SENDER_EMAIL != "your_email@gmail.com" and SENDER_PASSWORD != "your_app_password_here"
    })

@app.route('/api/test-email', methods=['POST'])
def test_email_api():
    """Test email endpoint"""
    try:
        data = request.json
        test_email = data.get('email')
        
        if not test_email:
            return jsonify({'success': False, 'message': 'No email provided'})
        
        success, msg = send_email(
            test_email,
            "Test Email - Smart Energy Platform",
            "This is a test email from Smart Energy Prediction Platform. If you received this, email is working correctly!"
        )
        
        return jsonify({
            'success': success,
            'message': msg
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ===================== MAIN ROUTE =====================

@app.route('/')
def index():
    """Main page - serves the HTML frontend"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/debug')
def debug_page():
    """Debug page"""
    conn = get_db_connection()
    user_count = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()
    prediction_count = conn.execute('SELECT COUNT(*) as count FROM predictions').fetchone()
    contact_count = conn.execute('SELECT COUNT(*) as count FROM contacts').fetchone()
    review_count = conn.execute('SELECT COUNT(*) as count FROM reviews').fetchone()
    conn.close()
    
    return f"""
    <html>
    <head>
        <title>Debug - Smart Energy Platform</title>
        <style>
            body {{ font-family: Arial; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #333; }}
            .info {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .warning {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .success {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            a {{ color: #4CAF50; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Energy Platform - Debug Info</h1>
            
            <div class="info">
                <h3>Database Status</h3>
                <p><strong>Users:</strong> {user_count['count']}</p>
                <p><strong>Predictions:</strong> {prediction_count['count']}</p>
                <p><strong>Contacts:</strong> {contact_count['count']}</p>
                <p><strong>Reviews:</strong> {review_count['count']}</p>
            </div>
            
            <div class="info">
                <h3>Email Configuration</h3>
                <p><strong>Sender Email:</strong> {SENDER_EMAIL}</p>
                <p><strong>Password Set:</strong> {'✅ Yes' if SENDER_PASSWORD and SENDER_PASSWORD != 'your_app_password_here' else '❌ No - Update in code!'}</p>
            </div>
            
            <div class="info">
                <h3>Service Status</h3>
                <p><strong>Model Loaded:</strong> {'✅ Yes' if model else '⚠️ No (using simulation)'}</p>
                <p><strong>Database:</strong> ✅ Connected</p>
                <p><strong>Session:</strong> {'✅ Active' if 'user_id' in session else '⚠️ Not logged in'}</p>
            </div>
            
            <div class="warning">
                <h3>Quick Actions</h3>
                <p><a href="/">Main Website</a></p>
                <p><a href="/api/health">Health Check</a></p>
                <p><button onclick="testEmail()">Test Email Function</button></p>
            </div>
            
            <div class="success">
                <h3>API Endpoints</h3>
                <ul>
                    <li><code>POST /api/register</code> - User registration</li>
                    <li><code>POST /api/login</code> - User login</li>
                    <li><code>POST /api/logout</code> - User logout</li>
                    <li><code>GET /api/check-auth</code> - Check authentication</li>
                    <li><code>POST /api/contact</code> - Contact form</li>
                    <li><code>POST /api/predict</code> - Energy prediction</li>
                    <li><code>GET /api/predictions</code> - Get predictions</li>
                    <li><code>POST /api/chatbot</code> - AI assistant</li>
                    <li><code>POST /api/review</code> - Submit review</li>
                    <li><code>GET /api/reviews</code> - Get reviews</li>
                    <li><code>POST /api/query</code> - Submit query</li>
                    <li><code>GET /api/gamification</code> - Get gamification data</li>
                    <li><code>GET /api/dashboard/stats</code> - Dashboard statistics</li>
                </ul>
            </div>
            
            <script>
                function testEmail() {{
                    const email = prompt("Enter email to test:");
                    if (email) {{
                        fetch('/api/test-email', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{email: email}})
                        }})
                        .then(response => response.json())
                        .then(data => {{
                            alert(data.success ? '✅ Test email sent!' : '❌ Failed: ' + data.message);
                        }});
                    }}
                }}
            </script>
        </div>
    </body>
    </html>
    """

# Create necessary directories
def setup_directories():
    """Create required directories"""
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    # Setup directories
    setup_directories()
    
    # Print startup info
    print("\n" + "="*60)
    print("SMART ENERGY PREDICTION PLATFORM")
    print("="*60)
    print(f"🌐 Server: http://localhost:5000")
    print(f"📧 Email: {SENDER_EMAIL}")
    print(f"🔧 Debug: http://localhost:5000/debug")
    print(f"🏥 Health: http://localhost:5000/api/health")
    
    # Check email configuration
    if SENDER_EMAIL == "your_email@gmail.com" or SENDER_PASSWORD == "your_app_password_here":
        print("\n⚠️  WARNING: Email not configured!")
        print("   Update SENDER_EMAIL and SENDER_PASSWORD in the code")
        print("   Get app password from: https://myaccount.google.com/apppasswords\n")
    
    print("="*60)
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
