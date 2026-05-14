from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
import hashlib
import urllib.request
import urllib.parse

app = Flask(__name__)
CORS(app)
app.secret_key = "heatguard_secret_key_2024"   # change in production

# ---------------- TELEGRAM CONFIG ----------------
BOT_TOKEN = "8610067144:AAFUB0jurgFPJtkkWyE2wvY692wMMC5fAr8"
CHAT_ID   = "5897872243"   # fallback chat ID

# ---------------- LOAD MODEL ----------------
model_data   = joblib.load("rf_model.pkl")
model        = model_data["model"]
scaler       = model_data["scaler"]
feature_cols = model_data["feature_cols"]
label_map    = model_data["label_map"]

history = []

# ---------------- DATABASE ----------------
DB_PATH = "heatguard.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            username TEXT    UNIQUE NOT NULL,
            phone    TEXT    NOT NULL,
            password TEXT    NOT NULL,
            created  TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialised → heatguard.db")

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, username, phone, password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "username": row[2],
                "phone": row[3], "password": row[4]}
    return None

# ---------------- HEAT INDEX ----------------
def calculate_heat_index(T, RH):
    T_f = (T * 9/5) + 32
    HI  = (-42.379
           + 2.04901523  * T_f
           + 10.14333127 * RH
           - 0.22475541  * T_f * RH
           - 0.00683783  * T_f**2
           - 0.05481717  * RH**2
           + 0.00122874  * T_f**2 * RH
           + 0.00085282  * T_f   * RH**2
           - 0.00000199  * T_f**2 * RH**2)
    return (HI - 32) * 5/9

# ---------------- TIME CATEGORY ----------------
def time_category(hour):
    if   5  <= hour < 12: return 0
    elif 12 <= hour < 17: return 1
    elif 17 <= hour < 21: return 2
    else:                 return 3

# ---------------- SEND TELEGRAM ----------------
def send_telegram(msg):
    try:
        payload = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text":    msg
        }).encode()
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        urllib.request.urlopen(url, data=payload, timeout=10)
        print(f"✅ Telegram sent")
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

# ================================================================
#  AUTH ROUTES
# ================================================================

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    username = data.get("username", "").strip()
    phone    = data.get("phone", "").strip()
    password = data.get("password", "").strip()

    if not all([name, username, phone, password]):
        return jsonify({"success": False, "error": "All fields required."})
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."})

    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "INSERT INTO users (name, username, phone, password, created) VALUES (?,?,?,?,?)",
            (name, username, phone, hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        print(f"✅ Registered: {username} ({phone})")
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Username already taken."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = get_user(username)
    if not user or user["password"] != hash_password(password):
        return jsonify({"success": False, "error": "Invalid username or password."})

    session["username"] = username
    session["name"]     = user["name"]
    session["phone"]    = user["phone"]
    print(f"✅ Login: {username}")
    return jsonify({"success": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/me")
def me():
    if "username" not in session:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "name":      session.get("name"),
        "username":  session.get("username"),
        "phone":     session.get("phone")
    })

# ================================================================
#  SENSOR DATA
# ================================================================

@app.route('/data', methods=['POST'])
def receive_data():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        ambient  = float(data['ambient_temp'])
        humidity = float(data['humidity'])
        body     = float(data['body_temp'])

        hour       = datetime.now().hour
        heat_index = calculate_heat_index(ambient, humidity)

        temp_humidity_stress = ambient * humidity / 100
        body_temp_deviation  = body - 37.0
        time_of_day          = time_category(hour)

        features = pd.DataFrame([[
            ambient, humidity, body, 30, hour,
            heat_index, temp_humidity_stress,
            body_temp_deviation, time_of_day
        ]], columns=feature_cols)

        features_scaled = pd.DataFrame(
            scaler.transform(features), columns=feature_cols
        )

        prediction = model.predict(features_scaled)[0]
        result     = label_map[prediction]

        record = {
            "time":       datetime.now().isoformat(),
            "ambient":    ambient,
            "humidity":   humidity,
            "body":       body,
            "heat_index": round(heat_index, 2),
            "risk":       result,
            "sms_sent":   False
        }

        history.append(record)
        if len(history) > 100:
            history.pop(0)

        print(f"✅ Received → Ambient:{ambient} Humidity:{humidity} Body:{body}")
        print(f"   Predicted → {result}")
        return jsonify(record)

    except Exception as e:
        print(f"❌ Error in /data: {e}")
        return jsonify({"error": str(e)}), 500

# ================================================================
#  EVALUATE — triggered by dashboard button, auto Telegram on High Risk
# ================================================================

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        if not history:
            return jsonify({"error": "No sensor data yet."}), 400

        record     = history[-1].copy()
        result     = record["risk"]
        alert_sent = False

        if "high" in result.lower():
            ts  = record["time"].split("T")[1][:8]

            risk_lower = result.lower()
            if "very high" in risk_lower or "extreme" in risk_lower or "danger" in risk_lower:
                action = (
                    "🔴 ACTION: Call emergency services immediately.\n"
                    "Move person to a cool area, apply cold water/ice packs\n"
                    "to neck, armpits & groin. Do NOT leave them alone."
                )
            else:
                action = (
                    "🟠 ACTION: Move to a shaded/cool area immediately.\n"
                    "Provide cold water to drink, loosen clothing,\n"
                    "apply wet cloth to skin & monitor closely."
                )

            msg = (
                f"🚨 HeatGuard ALERT\n"
                f"──────────────────\n"
                f"Risk Level : {result}\n"
                f"Body Temp  : {record['body']}°C\n"
                f"Ambient    : {record['ambient']}°C\n"
                f"Humidity   : {record['humidity']}%\n"
                f"Heat Index : {record['heat_index']}°C\n"
                f"Time       : {ts}\n"
                f"──────────────────\n"
                f"{action}\n"
                f"──────────────────\n"
                f"⚠ IMMEDIATE ATTENTION REQUIRED"
            )
            alert_sent = send_telegram(msg)
            if alert_sent:
                history[-1]['sms_sent'] = True
                record['sms_sent'] = True

        record['alert_sent'] = alert_sent
        return jsonify(record)

    except Exception as e:
        print(f"❌ Error in /evaluate: {e}")
        return jsonify({"error": str(e)}), 500

# ================================================================
#  MANUAL TELEGRAM (dashboard Send button)
# ================================================================

@app.route('/send_sms', methods=['POST'])
def send_sms():
    try:
        data = request.get_json()
        msg  = data.get('body', 'HeatGuard Alert!')
        ok   = send_telegram(msg)
        if ok and history:
            history[-1]['sms_sent'] = True
        return jsonify({'success': ok})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ================================================================
#  HISTORY / LATEST
# ================================================================

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history[::-1])

@app.route('/latest', methods=['GET'])
def latest():
    return jsonify(history[-1] if history else {})

# ================================================================
#  RUN
# ================================================================

if __name__ == "__main__":
    init_db()
    print("\n" + "="*50)
    print("  HeatGuard Flask Server")
    print(f"  Local IP : 192.168.43.180")
    print(f"  Login    : http://192.168.149.180:5000")
    print(f"  Endpoint : http://192.168.149.180:5000/data")
    print("="*50 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )