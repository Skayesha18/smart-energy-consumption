from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'smart-energy-ai-secret-key-2024'
CORS(app)  # Enable CORS for all routes

# =============================================
# EMAIL CONFIGURATION - UPDATE THESE!
# =============================================
SENDER_EMAIL = "teamsmartenergy12@gmail.com"  # Your email
SENDER_PASSWORD = "yzut aang kjfm sxud"    # Your Gmail app password
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

# In-memory storage (use database in production)
reviews = []
queries = []
predictions = []
contacts = []

# Feature names for the model
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'Occupancy', 'HVACUsage', 
    'LightingUsage', 'RenewableEnergy', 'DayOfWeek', 'Holiday',
    'Zscore', 'IsWeekend', 'Total_Device_Power', 'High_Load',
    'Rolling_Mean_3', 'Rolling_Mean_24'
]

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

def send_email(to_email, subject, body):
    """Send email with error handling"""
    try:
        # Check if email is configured
        if not SENDER_EMAIL or SENDER_EMAIL == "your_email@gmail.com":
            print("❌ Email not configured properly")
            return False, "Email service not configured"
        
        if not SENDER_PASSWORD:
            print("❌ Email password not set")
            return False, "Email password not set"
        
        print(f"📧 Attempting to send email to: {to_email}")
        print(f"   From: {SENDER_EMAIL}")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Try different SMTP methods
        try:
            # Method 1: SSL (port 465)
            print("   Trying SSL (port 465)...")
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ Email sent successfully via SSL to {to_email}")
            return True, "Email sent successfully"
            
        except Exception as e1:
            print(f"   SSL failed: {str(e1)[:100]}")
            
            try:
                # Method 2: TLS (port 587)
                print("   Trying TLS (port 587)...")
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

# ===================== ROUTES =====================

@app.route('/')
def index():
    """Main page - serves the HTML frontend"""
    return render_template('index.html')

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
        
        # Store contact
        contact_id = len(contacts) + 1
        contact = {
            'id': contact_id,
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        contacts.append(contact)
        
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
            # Still success but warn about email
            return jsonify({
                'success': True,
                'message': f'Message received! (Email notification failed: {email_message})',
                'contact_id': contact_id
            })
            
    except Exception as e:
        print(f"❌ Contact API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for energy prediction"""
    try:
        data = request.json
        
        # Get parameters with defaults
        temperature = float(data.get('temperature', 25))
        occupancy = int(data.get('occupancy', 2))
        humidity = float(data.get('humidity', 50))
        area = float(data.get('area', 1200))
        devices = data.get('devices', [])
        date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        time_str = data.get('time', '14:00')
        
        # Calculate hour for time factor
        hour = 14  # Default to 2 PM
        try:
            hour = int(time_str.split(':')[0])
        except:
            pass
        
        # Device power list - FIX: Define this BEFORE using it
        device_list = [
            ("Air Conditioner", 1500),
            ("Refrigerator", 150),
            ("LED TV", 100),
            ("Washing Machine", 500),
            ("Microwave", 1000),
            ("Laptop", 50),
            ("Desktop PC", 200),
            ("Lighting", 100),
            ("Electric Heater", 2000),
            ("Electric Oven", 2000)
        ]
        
        # Initialize device_power here - FIX: Initialize at start
        device_power = 0
        
        # Calculate device power
        for device_name in devices:
            for name, power in device_list:
                if isinstance(device_name, str) and device_name.lower() in name.lower():
                    device_power += power
                    break
        
        # Calculate prediction
        if model:
            # Use ML model if available
            features = {
                'temperature': temperature,
                'humidity': humidity,
                'occupancy': occupancy,
                'hvac': 'on' if 'Air Conditioner' in devices else 'off',
                'lighting': 'on' if 'Lighting' in devices else 'off',
                'renewable': 10,
                'day': datetime.now().weekday()
            }
            feature_array = engineer_features(features)
            if feature_array is not None:
                prediction = model.predict(feature_array)[0]
            else:
                prediction = 45.5
        else:
            # Simulate prediction
            # Base calculation
            base_consumption = (area / 100) * 0.5
            occupancy_factor = occupancy * 0.2
            temp_factor = abs(22 - temperature) * 0.15
            
            # Device factor calculation
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
        
        # Calculate additional metrics
        cost_per_kwh = 0.15 * 83  # INR conversion
        estimated_cost = prediction * cost_per_kwh
        carbon_footprint = prediction * 0.5
        
        # Calculate savings potential - FIX: Now device_power is always initialized
        savings_potential = min(30, (device_power / 5000) * 30)
        
        # Create prediction record
        prediction_id = len(predictions) + 1
        prediction_record = {
            'id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'date': date_str,
            'time': time_str,
            'temperature': temperature,
            'occupancy': occupancy,
            'humidity': humidity,
            'area': area,
            'devices': devices,
            'prediction': float(prediction),
            'cost': float(estimated_cost),
            'carbon': float(carbon_footprint),
            'savings_potential': float(savings_potential),
            'peak_hours': '6-9 PM' if 18 <= hour <= 21 else 'Normal Hours'
        }
        
        predictions.append(prediction_record)
        
        # Save to file for persistence
        try:
            with open('predictions.json', 'a') as f:
                json.dump(prediction_record, f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Could not save to file: {e}")
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'cost': float(estimated_cost),
            'carbon': float(carbon_footprint),
            'savings_potential': float(savings_potential),
            'unit': 'kWh',
            'currency': '₹',
            'prediction_id': prediction_id,
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Prediction failed: {str(e)}"}), 400

@app.route('/api/predictions', methods=['GET'])
def get_predictions_api():
    """Get recent predictions"""
    try:
        all_predictions = predictions.copy()
        
        # Load from file if exists
        if os.path.exists('predictions.json'):
            try:
                with open('predictions.json', 'r') as f:
                    for line in f:
                        if line.strip():
                            file_pred = json.loads(line.strip())
                            # Avoid duplicates
                            if not any(p.get('id') == file_pred.get('id') for p in all_predictions):
                                all_predictions.append(file_pred)
            except Exception as e:
                print(f"Warning reading predictions file: {e}")
        
        # Sort by timestamp
        all_predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Return last 20 predictions
        return jsonify(all_predictions[:20])
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify([])

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """Chatbot endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').lower().strip()
        
        responses = {
            'hello': "Hello! I'm your Smart Energy Assistant. How can I help you today?",
            'hi': "Hi there! Welcome to the Smart Energy Prediction Platform.",
            'help': "I can help with: 1) Energy predictions 2) Saving tips 3) Platform guidance 4) Technical questions",
            'predict': "Go to the Prediction section or tell me 'predict energy usage' for step-by-step guidance.",
            'contact': f"You can contact us at {SENDER_EMAIL} or use the Contact Us form.",
            'email': f"Our support email is {SENDER_EMAIL}",
            'project': "This is a Smart Energy Prediction Platform using AI to predict energy consumption and optimize usage.",
            'save': "To save energy: 1) Use efficient devices 2) Turn off unused appliances 3) Optimize temperature 4) Use natural light",
            'thank': "You're welcome! Feel free to ask more questions.",
            'bye': "Goodbye! Remember to check your energy predictions regularly."
        }
        
        response = "I can help with energy predictions, saving tips, or platform questions. What would you like to know?"
        
        for key in responses:
            if key in user_message:
                response = responses[key]
                break
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/review', methods=['POST'])
def submit_review_api():
    """Submit a review"""
    try:
        data = request.json
        review = {
            'id': len(reviews) + 1,
            'name': data.get('name', 'Anonymous'),
            'rating': int(data.get('rating', 5)),
            'comment': data.get('comment', ''),
            'timestamp': datetime.now().isoformat()
        }
        reviews.append(review)
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your review!',
            'review': review
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/reviews', methods=['GET'])
def get_reviews_api():
    """Get recent reviews"""
    return jsonify(reviews[-10:])

@app.route('/api/query', methods=['POST'])
def submit_query_api():
    """Submit a query"""
    try:
        data = request.json
        query = {
            'id': len(queries) + 1,
            'email': data.get('email', ''),
            'query': data.get('query', ''),
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        queries.append(query)
        
        # Send email notification
        email_body = f"""
New Query Received:

Email: {query['email']}
Query: {query['query']}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Query ID: #{query['id']}

Please respond within 24 hours.
        """
        
        # Send to admin email
        send_email(
            SENDER_EMAIL,
            f"New Query #{query['id']} - Smart Energy Platform",
            email_body
        )
        
        return jsonify({
            'success': True,
            'message': 'Query submitted successfully! We will respond within 24 hours.',
            'query_id': query['id']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Energy Prediction Platform',
        'timestamp': datetime.now().isoformat(),
        'predictions_count': len(predictions),
        'reviews_count': len(reviews),
        'queries_count': len(queries),
        'contacts_count': len(contacts),
        'email_configured': SENDER_EMAIL != "your_email@gmail.com" and SENDER_PASSWORD != "your_app_password_here"
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/debug')
def debug_page():
    """Debug page"""
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
            .error {{ background: #ffebee; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .success {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            a {{ color: #4CAF50; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Energy Platform - Debug Info</h1>
            
            <div class="info">
                <h3>Email Configuration</h3>
                <p><strong>Sender Email:</strong> {SENDER_EMAIL}</p>
                <p><strong>Password Set:</strong> {'✅ Yes' if SENDER_PASSWORD and SENDER_PASSWORD != 'your_app_password_here' else '❌ No - Update in code!'}</p>
            </div>
            
            <div class="info">
                <h3>Service Status</h3>
                <p><strong>Model Loaded:</strong> {'✅ Yes' if model else '⚠️ No (using simulation)'}</p>
                <p><strong>Total Predictions:</strong> {len(predictions)}</p>
                <p><strong>Total Queries:</strong> {len(queries)}</p>
                <p><strong>Total Reviews:</strong> {len(reviews)}</p>
                <p><strong>Total Contacts:</strong> {len(contacts)}</p>
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
                    <li><code>POST /api/contact</code> - Contact form</li>
                    <li><code>POST /api/predict</code> - Energy prediction</li>
                    <li><code>GET /api/predictions</code> - Get predictions</li>
                    <li><code>POST /api/chatbot</code> - AI assistant</li>
                    <li><code>POST /api/review</code> - Submit review</li>
                    <li><code>POST /api/query</code> - Submit query</li>
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
    
    # Create favicon if not exists
    favicon_path = 'static/favicon.ico'
    if not os.path.exists(favicon_path):
        try:
            # Create empty favicon file
            with open(favicon_path, 'wb') as f:
                pass
        except:
            pass

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
    
    app.run(debug=True, port=5000, host='0.0.0.0')