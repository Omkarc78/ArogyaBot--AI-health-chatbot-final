# Standard library imports
import asyncio
import csv
import os
import random
import string
import tempfile
from datetime import datetime, timedelta
# Third-party imports
import joblib
import pytz
import xgboost
from flask import Flask, render_template, request, redirect, url_for, flash, abort, send_file, make_response, jsonify
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import re


# Local application imports
from models import db, User, UserProfile, DiagnosisRecord, DoctorVisit, DoctorInfo
# from matching import extract_symptoms

# -------------------------
# CUSTOM TRAINED CHATBOT (LAZY LOADING)
# -------------------------
chat_tokenizer = None
chat_model = None
device = None
MODEL_PATH = "medical-chatbot-rlhf-jsonl"

def load_chat_model_once():
    """Loads the chatbot model and tokenizer on first use to avoid import errors in CLI."""
    global chat_tokenizer, chat_model, device
    if chat_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            print(f"CRITICAL ERROR: Failed to import chatbot dependencies. Try running: pip install --upgrade huggingface-hub transformers\nDetails: {e}")
            raise e
        print("🔵 Loading chatbot model once...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chat_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        chat_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        chat_model.eval()  # Set model to evaluation mode for faster, deterministic inference
        chat_tokenizer.pad_token = chat_tokenizer.eos_token



app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    # Change int(user_id) to just user_id (since id is now a string)
    return db.session.get(User, user_id)

# Create database tables
with app.app_context():
    db.create_all()

def admin_blocked_page():
    return render_template('access_denied.html', message="Admins are not allowed to access this page."), 403

DISEASE_SPECIALTY_MAP = {
    "fever": ["General", "Physician"],
    "viral": ["General", "Physician"],
    "malaria": ["Physician"],
    "headache": ["Neurologist"],
    "migraine": ["Neurologist"],
    "skin": ["Dermatologist"],
    "allergy": ["Dermatologist"],
    "heart": ["Cardiologist"],
}

def recommend_doctors(disease, severity, limit=3):
    """
    Recommends doctors based on disease and severity.
    - For High/Extreme severity, prioritizes specialists.
    - For Low/Moderate severity, includes general physicians.
    - Returns a randomized list of doctors.
    """
    if not disease:
        return DoctorInfo.query.order_by(db.func.random()).limit(limit).all()

    disease_lower = disease.lower()
    all_specialties = set()
    for key, specs in DISEASE_SPECIALTY_MAP.items():
        if key in disease_lower:
            for s in specs:
                all_specialties.add(s)

    if not all_specialties:
        # No mapping, return any doctors, randomized
        return DoctorInfo.query.order_by(db.func.random()).limit(limit).all()

    query = DoctorInfo.query
    specialist_specialties = [s for s in all_specialties if s.lower() not in ['general', 'physician']]

    if severity in ['High', 'Extreme'] and specialist_specialties:
        query = query.filter(db.or_(*[DoctorInfo.specialty.ilike(f"%{s}%") for s in specialist_specialties]))
    else:
        query = query.filter(db.or_(*[DoctorInfo.specialty.ilike(f"%{s}%") for s in all_specialties]))

    return query.order_by(db.func.random()).limit(limit).all()

@app.route('/book-appointment/<doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    from models import DoctorInfo, DoctorVisit

    doctor = DoctorInfo.query.filter_by(doctor_id=doctor_id).first_or_404()

    if request.method == 'POST':
        disease = request.form.get('disease', '').strip()
        notes = request.form.get('notes', '').strip()
        appointment_datetime_str = request.form.get('appointment_datetime')

        # --- Start of new validation ---
        if not disease:
            flash("Disease / Reason for Visit is required.", "danger")
            return render_template('book_appointment.html', doctor=doctor, user=current_user)

        if not appointment_datetime_str:
            flash("Please select an appointment date and time.", "danger")
            return render_template('book_appointment.html', doctor=doctor, user=current_user)

        try:
            appointment_datetime = datetime.fromisoformat(appointment_datetime_str)
        except ValueError:
            flash("Invalid date format. Please use the date picker.", "danger")
            return render_template('book_appointment.html', doctor=doctor, user=current_user)

        # Check for recent duplicate booking
        india_tz = pytz.timezone('Asia/Kolkata')
        twenty_four_hours_ago = datetime.now(india_tz) - timedelta(hours=24)
        recent_visit = DoctorVisit.query.filter(
            DoctorVisit.user_id == current_user.id,
            DoctorVisit.doctor_id == doctor.doctor_id,
            DoctorVisit.booked_at > twenty_four_hours_ago
        ).first()

        if recent_visit:
            flash("You have already booked an appointment with this doctor in the last 24 hours.", "warning")
            return redirect(url_for('home'))
        # --- End of new validation ---

        visit = DoctorVisit(
            user_id=current_user.id,
            doctor_id=doctor.doctor_id,
            disease=disease,
            notes=notes,
            appointment_datetime=appointment_datetime,
            booked_at=datetime.now(india_tz)
        )

        db.session.add(visit)
        db.session.commit()

        flash("Appointment booked successfully!", "success")
        return redirect(url_for('home'))

    return render_template(
        'book_appointment.html',
        doctor=doctor,
        user=current_user
    )






@app.route('/')
def home():
    if current_user.is_authenticated and getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            print(f"User {user.username} (ID: {user.id}) logged in successfully.")  # Debug print
            # --- Doctor Info Auto-Insert Logic ---
            if getattr(user, 'is_admin', 0) == 1:
                from models import DoctorInfo
                doctor = DoctorInfo.query.filter_by(doctor_id=str(user.id)).first()
                if not doctor:
                    new_doctor = DoctorInfo(
                        doctor_id=str(user.id),
                        full_name=user.full_name or user.username,
                        clinic='',
                        address='',
                        specialty=''
                        # Removed profile_photo=user.profile_photo or ''
                    )
                    db.session.add(new_doctor)
                    db.session.commit()
                return redirect(url_for('doctor'))
            return redirect(url_for('home'))
        else:
            error = "Invalid email or password."
    return render_template('login.html', error=error)

def generate_custom_user_id():
    """
    Generate a unique user ID: 4 uppercase letters + 6 digits (e.g., 'ABCD123456').
    Ensures uniqueness by checking the database (even with concurrent requests).
    """
    while True:
        prefix = ''.join(random.choices(string.ascii_uppercase, k=4))
        digits = ''.join(random.choices(string.digits, k=6))
        custom_id = f"{prefix}{digits}"
        # Use db.session.get() for primary key lookup (fastest, avoids race)
        if not db.session.get(User, custom_id):
            return custom_id

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        mobile = request.form['mobile']
        password = request.form['password']

        # Defensive: Check for empty username/email after stripping
        if not username:
            error = "Username cannot be empty."
        elif not email:
            error = "Email cannot be empty."
        else:
            user_by_username = User.query.filter_by(username=username).first()
            user_by_email = User.query.filter_by(email=email).first()
            if user_by_username:
                error = "Username already exists. Please choose another."
            elif user_by_email:
                error = "Email already exists. Please use another."
            else:
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                # The generate_custom_user_id function already ensures the ID is unique.
                custom_id = generate_custom_user_id()

                new_user = User(
                    id=custom_id,
                    full_name=full_name,
                    username=username,
                    email=email,
                    mobile=mobile,
                    password=hashed_password
                )
                db.session.add(new_user)
                try:
                    db.session.commit()
                    return redirect(url_for('login'))
                except Exception as e:
                    db.session.rollback()
                    error = "An error occurred while creating your account. Please try again."
    return render_template('signup.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))






# Load models once at startup
# Global variables for the models, to be lazy-loaded
mlb_encoder = None
label_encoder = None
model = None

def load_disease_model_once():
    """Loads the disease prediction models on first use to avoid CLI warnings."""
    global mlb_encoder, label_encoder, model
    if model is None:
        # Make sure the paths are correct and point to your actual files
        mlb_encoder = joblib.load("trained_model/xg_boost/mlb_encoder.pkl")
        label_encoder = joblib.load("trained_model/xg_boost/label_encoder.pkl")
        model = joblib.load("trained_model/xg_boost/xgb_model.pkl")

def predict_disease_from_symptoms_fallback(symptoms):
    """
    Fallback prediction using rule-based mapping from CSV diseases.
    Loads diseases from symptom_precaution.csv dynamically.
    Maps common symptom sets to diseases (expand with more mappings as needed).
    """
    import csv
    from io import StringIO  # For testing; use file path in production
    
    # Load diseases from CSV (replace with file path)
    csv_path = "static/data/symptom_precaution.csv"  # Relative to app
    diseases = []
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            diseases = [row['Disease'].strip() for row in reader]
    except FileNotFoundError:
        # Fallback to hardcoded from provided snippet
        diseases = ['Drug Reaction', 'Malaria', 'Allergy', 'Hypothyroidism']
    
    # Rule-based mapping (common symptoms → disease; expand based on full CSV/medical knowledge)
    combo_map = {
        frozenset(['skin_rash', 'itching', 'swelling']): 'Drug Reaction',
        frozenset(['high_fever', 'chills', 'headache']): 'Malaria',  # Note: Order doesn't matter due to frozenset
        frozenset(['itching', 'skin_rash', 'runny_nose']): 'Allergy',
        frozenset(['fatigue', 'weight_gain', 'cold_intolerance']): 'Hypothyroidism',
        # Auto-add more: e.g., for fever/headache/nausea (common query) → Malaria or Allergy
        frozenset(['fever', 'headache', 'nausea']): 'Malaria',  # Example extension
        # TODO: Dynamically expand by parsing full CSV for symptom hints if available
    }
    
    # Normalize input symptoms (lowercase)
    norm_symptoms = frozenset([s.strip().lower() for s in symptoms if s.strip()])
    return combo_map.get(norm_symptoms, '')  # Return matching disease or empty

def predict_disease_from_symptoms(symptoms_list):
    # Lazy-load the models to prevent loading during CLI commands
    load_disease_model_once()

    if not symptoms_list:
        return ""
    if isinstance(symptoms_list, str):
        symptoms_list = [s.strip() for s in symptoms_list.split(",") if s.strip()]
    # Enhanced: Lowercase, replace spaces/hyphens with _, try common variants
    normalized = []
    for s in symptoms_list:
        s_clean = s.strip().lower()
        s_clean = re.sub(r'[\s\-_]+', '_', s_clean)  # Spaces, hyphens → _
        # Common fixes: e.g., 'head ache' → 'headache'
        if 'head ache' in s_clean:
            s_clean = s_clean.replace('head_ache', 'headache')
        normalized.append(s_clean)
    print(f"Normalized symptoms: {normalized}")  # Debug
    
    try:
        X_input = mlb_encoder.transform([normalized])
        y_pred_encoded = model.predict(X_input)
        predicted_disease = label_encoder.inverse_transform(y_pred_encoded)[0]
        print(f"Predicted disease: {predicted_disease}")  # Debug
        return predicted_disease
    except Exception as e:
        print("Prediction error:", e)
        # Fallback: Try without normalization or common symptoms
        fallback = predict_disease_from_symptoms_fallback(normalized)  # Define below
        return fallback or ""

def get_disease_details(disease_name):
    """
    Fetch disease details, diet, exercise, medicine, and precautions from the CSV.
    Returns a dict with keys: description, diet, exercise, medicine, precautions (list)
    """
    details = {
        "description": "",
        "diet": "",
        "exercise": "",
        "medicine": "",
        "precautions": []
    }
    try:
        # Use encoding='utf-8-sig' and errors='replace' to avoid UnicodeDecodeError
        with open("static/data/symptom_precaution.csv", newline='', encoding='utf-8-sig', errors='replace') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Disease'].strip().lower() == disease_name.strip().lower():
                    details["description"] = row.get("Disease_des", "")
                    details["diet"] = row.get("diet", "")
                    details["exercise"] = row.get("Exercise", "")
                    details["medicine"] = row.get("medicine", "")
                    details["precautions"] = [
                        row.get("Precaution_1", ""),
                        row.get("Precaution_2", ""),
                        row.get("Precaution_3", ""),
                        row.get("Precaution_4", "")
                    ]
                    break
    except FileNotFoundError:
        print("Warning: symptom_precaution.csv not found. Disease details will be empty.")
    return details

@app.route('/diagnose', methods=['GET', 'POST'])
@login_required
def upload():
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    show_extra_fields = False
    error = None
    symptoms = ''
    de_symptoms = ''
    predicted_disease = ''
    severity = ''
    diet_plan = ''
    exercise_plan = ''
    medicine = ''
    past_records = DiagnosisRecord.query.filter_by(
        user_id=current_user.id
    ).order_by(DiagnosisRecord.diagnosed_at.desc()).all()

    if request.method == 'POST' and request.form.get('submit') == 'save':
        # Get selected keys and values from hidden fields
        selected_keys = request.form.get('symptoms', '').strip()
        selected_values = request.form.get('de_symptoms', '').strip()
        if not selected_keys or not selected_values:
            error = "Please select at least one symptom from the suggestions."
            flash(error, "danger")
            return render_template(
                'diagnose.html',
                show_extra_fields=False,
                symptoms='',
                de_symptoms='',
                predicted_disease='',
                severity='',
                diet_plan='',
                exercise_plan='',
                medicine='',
                past_records=past_records,
                disease_details={},
                error=error
            )
        symptoms = selected_keys
        de_symptoms = selected_values
        matched_symptoms = [s.strip() for s in selected_values.split(',') if s.strip()]
        india_tz = pytz.timezone('Asia/Kolkata')
        diagnosed_at = datetime.now(india_tz)
        has_history = DiagnosisRecord.query.filter_by(user_id=current_user.id).first() is not None
        predicted_disease = predict_disease_from_symptoms(matched_symptoms)
        severity_sum, risk_level = assess_severity(matched_symptoms)
        severity = risk_level  # Store only the risk_level string in DB
        recommendations = generate_recommendations(predicted_disease)
        disease_details = get_disease_details(predicted_disease)
        # Set medicine from disease_details, not recommendations
        new_record = DiagnosisRecord(
            user_id=current_user.id,
            symptoms=symptoms,
            de_symptoms=de_symptoms,
            diagnosed_at=diagnosed_at,
            used_history=int(has_history),
            predicted_disease=predicted_disease,
            severity=severity,  # This is the value stored in the DB
            diet_plan=disease_details.get('diet', ''),
            exercise_plan=disease_details.get('exercise', ''),
            medicine=disease_details.get('medicine', '')
        )
        try:
            db.session.add(new_record)
            db.session.commit()
            flash("Diagnosis completed successfully!", "success")
            return redirect(url_for('upload', show_latest='1'))
        except Exception as e:
            db.session.rollback()
            error = f"Error saving record: {str(e)}"
            flash(error, "danger")
            return render_template(
                'diagnose.html',
                show_extra_fields=False,
                symptoms=symptoms,
                de_symptoms=de_symptoms,
                predicted_disease=predicted_disease,
                severity=severity,
                diet_plan=diet_plan,
                exercise_plan=exercise_plan,
                medicine=medicine,
                past_records=past_records,
                disease_details=disease_details,
                error=error
            )

    if request.args.get('show_latest') == '1' and past_records:
        latest = past_records[0]
        symptoms = ''  # Clear textarea after save or refresh
        de_symptoms = latest.de_symptoms
        show_extra_fields = True
        predicted_disease = latest.predicted_disease
        # Use the stored severity string directly
        severity = latest.severity  # This is the value fetched from the DB
        diet_plan = latest.diet_plan
        exercise_plan = latest.exercise_plan
        medicine = latest.medicine
        disease_details = get_disease_details(predicted_disease)
    else:
        disease_details = {}

    return render_template(
        'diagnose.html',
        show_extra_fields=show_extra_fields,
        symptoms=symptoms,
        de_symptoms=de_symptoms,
        predicted_disease=predicted_disease,
        severity=severity,  # This is passed to the template
        diet_plan=diet_plan,
        exercise_plan=exercise_plan,
        medicine=medicine,
        past_records=past_records,
        disease_details=disease_details,
        error=error
    )

# Example prediction functions (replace with your actual implementations)
def assess_severity(symptoms_list):
    """
    Calculate severity as the sum of weights for all symptoms in symptoms_list,
    using static/data/Symptom-severity.csv.
    Returns (severity_sum, risk_level)
    """
    if not symptoms_list:
        return 0, "Low"
    severity_map = {}
    try:
        # Use encoding='utf-8-sig' and errors='replace' to avoid UnicodeDecodeError
        with open("static/data/Symptom-severity.csv", newline='', encoding='utf-8-sig', errors='replace') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Lowercase for robust matching
                severity_map[row['Symptom'].strip().lower()] = int(row['weight'])
    except FileNotFoundError:
        print("Warning: Symptom-severity.csv not found. Severity assessment will default to 'Low'.")
        return 0, "Low"
    except Exception as e:
        print(f"Error reading Symptom-severity.csv: {e}")
        return 0, "Low"
    total = 0
    for s in symptoms_list:
        weight = severity_map.get(s.strip().lower(), 0)
        total += weight
    # Updated severity ranges
    if total > 33:
        risk_level = "Extreme"
    elif total > 22:
        risk_level = "High"
    elif total > 12:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    return total, risk_level

def generate_recommendations(disease):
    """Your recommendation generation logic goes here"""
    return {
        'diet': "Balanced diet with...",
        'exercise': "30 minutes of...", 
        'medicine': "Consult doctor for..."
    }


@app.route('/history')
@login_required
def history():
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    # Fetch all diagnosis records for the current user
    past_records = DiagnosisRecord.query.filter_by(
        user_id=current_user.id
    ).order_by(DiagnosisRecord.diagnosed_at.desc()).all()
    return render_template(
        'medical_history.html',
        past_records=past_records,
        assess_severity=assess_severity,
        get_disease_details=get_disease_details
    )


@app.route('/profile', methods=['GET'])
@login_required
def profile():
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    # List all images in static/profile/
    profile_photos = []
    profile_folder = os.path.join(app.static_folder, 'profile')
    if os.path.exists(profile_folder):
        profile_photos = [f for f in os.listdir(profile_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg'))]
    return render_template("profile.html", profile=profile, profile_photos=profile_photos)

@app.route('/profile/change_photo', methods=['POST'])
@login_required
def change_profile_photo():
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    photo_path = request.form.get('profile_photo')
    upload_file = request.files.get('upload_profile_photo')
    # Handle uploaded file
    if upload_file and upload_file.filename:
        # Save to static/profile/persnol_profile/
        upload_folder = os.path.join(app.static_folder, 'profile', 'persnol_profile')
        os.makedirs(upload_folder, exist_ok=True)
        filename = upload_file.filename
        save_path = os.path.join(upload_folder, filename)
        upload_file.save(save_path)
        # Store relative path for use in src
        photo_path = f'static/profile/persnol_profile/{filename}'
    # Update UserProfile and User
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if profile:
        profile.profile_photo = photo_path
    current_user.profile_photo = photo_path
    db.session.commit()
    return redirect(url_for('profile'))

@app.route('/profile/edit/<section>', methods=['GET', 'POST'])
@login_required
def edit_profile(section):
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.session.add(profile)
        db.session.commit()
    if request.method == 'POST':
        if section == 'basic':
            profile.full_name = request.form['full_name']
            profile.profile_photo = request.form['profile_photo']
            profile.age = request.form['age']
        else:
            profile.age = request.form['age']
            profile.height = request.form['height']
            profile.weight = request.form['weight']
            profile.blood_group = request.form['blood_group']
            profile.gender = request.form['gender']
            profile.chronic_conditions = request.form['chronic_conditions']
            profile.allergies = request.form['allergies']
            profile.notes = request.form['notes']
        profile.updated_at = datetime.utcnow()
        db.session.commit()
        return redirect(url_for('profile'))
    return render_template("edit_profile.html", profile=profile, section=section)

@app.route('/profile/edit/basic', methods=['POST'])
@login_required
def edit_profile_basic():
    if getattr(current_user, 'is_admin', 0) == 1:
        return admin_blocked_page()
    full_name = request.form['full_name']
    username = request.form['username']
    email = request.form['email']
    mobile = request.form['mobile']
    emergency_contact_name = request.form.get('emergency_contact_name', '')
    emergency_relationship = request.form.get('emergency_relationship', '')
    emergency_contact_mobile = request.form.get('emergency_contact_mobile', '')
    current_user.full_name = full_name
    current_user.username = username
    current_user.email = email
    current_user.mobile = mobile
    current_user.emergency_contact_name = emergency_contact_name
    current_user.emergency_relationship = emergency_relationship
    current_user.emergency_contact_mobile = emergency_contact_mobile
    db.session.commit()
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if profile:
        profile.full_name = full_name
        db.session.commit()

    return redirect(url_for('profile'))


@app.route('/my_card')
@login_required
def my_card():
    return render_template('my_card.html', user=current_user)

@app.route('/download_my_card_png')
@login_required
def download_my_card_png():
    return render_template('my_card.html', user=current_user)

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/doctor/appointments')
@login_required
def doctor_appointments():
    visits = DoctorVisit.query.filter_by(
        doctor_id=str(current_user.id)
    ).order_by(DoctorVisit.booked_at.desc()).all()
    return render_template('doctor_appointments.html', visits=visits)

@app.route('/doctor/appointment/<int:id>/confirm', methods=['POST'])
@login_required
def confirm_appointment(id):
    visit = DoctorVisit.query.get_or_404(id)
    # Ensure the current doctor is the one assigned to this visit
    if str(visit.doctor_id) != str(current_user.id):
        abort(403)
    visit.status = 'confirmed'
    db.session.commit()
    flash(f"Appointment for patient {visit.user_id} has been confirmed.", "success")

    # Redirect back to the patient detail page if user_id is provided, else to the appointments list
    patient_user_id = request.args.get('user_id')
    if patient_user_id:
        return redirect(url_for('doctor', user_id=patient_user_id))
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor/appointment/<int:id>/reject', methods=['POST'])
@login_required
def reject_appointment(id):
    visit = DoctorVisit.query.get_or_404(id)
    # Ensure the current doctor is the one assigned to this visit
    if str(visit.doctor_id) != str(current_user.id):
        abort(403)
    original_status = visit.status
    visit.status = 'rejected'
    db.session.commit()
    if original_status == 'pending':
        flash(f"Appointment for patient {visit.user_id} has been rejected.", "info")
    else:
        flash(f"Appointment for patient {visit.user_id} has been cancelled.", "warning")

    # Redirect back to the patient detail page if user_id is provided, else to the appointments list
    patient_user_id = request.args.get('user_id')
    if patient_user_id:
        return redirect(url_for('doctor', user_id=patient_user_id))
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor', methods=['GET', 'POST'])
@login_required
def doctor():
    if not getattr(current_user, 'is_admin', 0):
        return redirect(url_for('home'))
    
    from models import DoctorInfo, User, UserProfile, DiagnosisRecord, DoctorVisit
    doctor = DoctorInfo.query.filter_by(doctor_id=str(current_user.id)).first()
    user = None
    profile = None
    diagnosis_records = []
    
    # Handle patient search (GET) - replaces admin dashboard functionality
    patient_id = request.args.get('user_id', '').strip()
    if patient_id:
        user = User.query.filter_by(id=patient_id, is_admin=False).first()
        if user:
            profile = UserProfile.query.filter_by(user_id=user.id).first()
            if not profile:
                profile = UserProfile(user_id=user.id)
                db.session.add(profile)
                db.session.commit()
            diagnosis_records = DiagnosisRecord.query.filter_by(user_id=user.id).order_by(DiagnosisRecord.diagnosed_at.desc()).all()
        else:
            flash("No user found with this Patient ID.", "danger")

    # Handle POST requests for all forms
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        if form_type == 'doctor_visit':
            # Add doctor visit record
            disease = request.form.get('disease', '').strip()
            blood_report = request.form.get('blood_report', '').strip()
            suggested_medicine = request.form.get('suggested_medicine', '').strip()
            notes = request.form.get('notes', '').strip()
            user_id = request.form.get('user_id', '').strip()
            doctor_id = str(current_user.id)
            india_tz = pytz.timezone('Asia/Kolkata')
            booked_at_time = datetime.now(india_tz)
            
            appointment_datetime_str = request.form.get('appointment_datetime')
            appointment_datetime = datetime.fromisoformat(appointment_datetime_str) if appointment_datetime_str else booked_at_time
            
            if user_id and doctor_id:
                new_visit = DoctorVisit(
                    disease=disease,
                    blood_report=blood_report,
                    suggested_medicine=suggested_medicine,
                    notes=notes,
                    user_id=user_id,
                    doctor_id=doctor_id,
                    appointment_datetime=appointment_datetime,
                    booked_at=booked_at_time
                )
                db.session.add(new_visit)
                db.session.commit()
                flash("Doctor visit record added successfully.", "success")
            return redirect(url_for('doctor', user_id=user_id))
            
        elif form_type == 'edit_patient_info':
            # Update patient profile information
            user_id = request.args.get('user_id', '').strip() or request.form.get('user_id', '').strip()
            if user_id:
                profile = UserProfile.query.filter_by(user_id=user_id).first()
                if profile:
                    profile.age = request.form.get('age')
                    profile.height = request.form.get('height')
                    profile.weight = request.form.get('weight')
                    profile.gender = request.form.get('gender')
                    profile.blood_group = request.form.get('blood_group')
                    profile.chronic_conditions = request.form.get('chronic_conditions')
                    profile.allergies = request.form.get('allergies')
                    profile.current_medication = request.form.get('current_medication')
                    profile.notes = request.form.get('notes')
                    profile.updated_at = datetime.utcnow()
                    db.session.commit()
                    flash("Patient info updated successfully.", "success")
            return redirect(url_for('doctor', user_id=user_id))
            
        elif form_type == 'update_patient_details':
            # Update patient details (replaces admin_update functionality)
            user_id = request.form.get('user_id', '').strip()
            if user_id:
                user = User.query.get_or_404(user_id)
                profile = UserProfile.query.filter_by(user_id=user.id).first()
                if not profile:
                    profile = UserProfile(user_id=user.id)
                    db.session.add(profile)
                    db.session.commit()
                
                profile.age = request.form.get('age')
                profile.gender = request.form.get('gender')
                profile.height = request.form.get('height')
                profile.weight = request.form.get('weight')
                profile.blood_group = request.form.get('blood_group')
                profile.chronic_conditions = request.form.get('chronic_conditions')
                profile.allergies = request.form.get('allergies')
                profile.current_medication = request.form.get('current_medication')
                profile.notes = request.form.get('notes')
                profile.updated_at = datetime.utcnow()
                db.session.commit()
                flash("Patient details updated successfully.", "success")
                return redirect(url_for('doctor', user_id=user_id))

    return render_template('doctor.html', doctor=doctor, user=user, profile=profile, 
                         diagnosis_records=diagnosis_records, pytz=pytz)

@app.route('/doctor/delete_diagnosis_record', methods=['POST'])
@login_required
def delete_diagnosis_record():
    if not getattr(current_user, 'is_admin', 0):
        return jsonify({'success': False}), 403
        
    data = request.get_json()
    record_id = data.get('id')
    record = DiagnosisRecord.query.get(record_id)
    if record:
        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

@app.route('/doctor/change_photo', methods=['POST'])
@login_required
def doctor_change_photo():
    if not getattr(current_user, 'is_admin', 0):
        return redirect(url_for('home'))
        
    from models import DoctorInfo
    doctor = DoctorInfo.query.filter_by(doctor_id=str(current_user.id)).first()
    upload_file = request.files.get('upload_profile_photo')
    
    if upload_file and upload_file.filename:
        upload_folder = os.path.join(app.static_folder, 'profile', 'doctor_profile')
        os.makedirs(upload_folder, exist_ok=True)
        filename = upload_file.filename
        save_path = os.path.join(upload_folder, filename)
        upload_file.save(save_path)
        photo_path = f'profile/doctor_profile/{filename}'
        
        user = User.query.get(current_user.id)
        if user:
            user.profile_photo = photo_path
        db.session.commit()
        flash("Profile photo updated successfully.", "success")
    else:
        flash("No photo selected.", "danger")
    return redirect(url_for('doctor'))

@app.route('/doctor/change_clinic_photo', methods=['POST'])
@login_required
def doctor_change_clinic_photo():
    if not getattr(current_user, 'is_admin', 0):
        return redirect(url_for('home'))
        
    from models import DoctorInfo
    doctor = DoctorInfo.query.filter_by(doctor_id=str(current_user.id)).first()
    upload_file = request.files.get('upload_clinic_photo')
    
    if upload_file and upload_file.filename:
        upload_folder = os.path.join(app.static_folder, 'profile', 'clinic_photo')
        os.makedirs(upload_folder, exist_ok=True)
        filename = upload_file.filename
        save_path = os.path.join(upload_folder, filename)
        upload_file.save(save_path)
        photo_path = f'profile/clinic_photo/{filename}'
        
        if doctor:
            doctor.clinic_photo = photo_path
        db.session.commit()
        flash("Clinic photo updated successfully.", "success")
    else:
        flash("No photo selected.", "danger")
    return redirect(url_for('doctor'))

@app.route('/doctor/edit_info', methods=['POST'])
@login_required
def doctor_edit_info():
    if not getattr(current_user, 'is_admin', 0):
        return redirect(url_for('home'))
        
    from models import DoctorInfo, User
    doctor = DoctorInfo.query.filter_by(doctor_id=str(current_user.id)).first()
    user = User.query.get(current_user.id)
    
    full_name = request.form.get('full_name', '').strip()
    clinic = request.form.get('clinic', '').strip()
    address = request.form.get('address', '').strip()
    specialty = request.form.get('specialty', '').strip()
    
    if doctor:
        doctor.full_name = full_name
        doctor.clinic = clinic
        doctor.address = address
        doctor.specialty = specialty
    if user:
        user.full_name = full_name
        
    db.session.commit()
    flash("Doctor info updated successfully.", "success")
    return redirect(url_for('doctor'))

@app.route('/footer')
def footer():
    return render_template('footer.html')




def chat_with_model(user_query):
    try:
        load_chat_model_once()

        system_instruction = (
            "You are a medical assistant. Respond clearly and safely."
        )

        prompt = f"{system_instruction}\nUser: {user_query}\nBot:"

        inputs = chat_tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = chat_model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        raw_output = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = raw_output.split("Bot:")[-1].strip()

        return reply or "I understood your symptoms. Please consult a doctor."

    except Exception as e:
        print("CHATBOT ERROR:", e)
        return "⚠️ AI service is temporarily unavailable. Please try again later."



@app.route('/chatbot_api', methods=['POST'])
@login_required
def chatbot_api():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Existing chatbot response
    bot_reply = chat_with_model(user_message) or "No response generated."


    # Extract symptoms from user_message (e.g., "with fever, headache, nausea" -> ['fever', 'headache', 'nausea'])
    def extract_symptoms(query):
        match = re.search(r'with\s+(.+?)(?:\.$|\s*$)', query, re.IGNORECASE)
        if match:
            symptoms_str = match.group(1).strip()
            return [s.strip() for s in symptoms_str.split(',') if s.strip()]
        return []

    extracted_symptoms = extract_symptoms(user_message)
    
    # Get last diagnosis as fallback
    last_diagnosis = (
        DiagnosisRecord.query
        .filter_by(user_id=current_user.id)
        .order_by(DiagnosisRecord.id.desc())
        .first()
    )
    
    # Compute fresh prediction if symptoms extracted, else fallback to last
    if extracted_symptoms:
        predicted_disease = predict_disease_from_symptoms(extracted_symptoms)
        severity_sum, severity = assess_severity(extracted_symptoms)
    else:
        predicted_disease = last_diagnosis.predicted_disease if last_diagnosis else ""
        severity = last_diagnosis.severity if last_diagnosis else "Low"
    
    # Recommend doctors based on computed disease
    recommended_doctors = recommend_doctors(predicted_disease, severity)
    print(f"Extracted: {extracted_symptoms}, Predicted: {predicted_disease}")
    # Save chatbot conversation with computed values
    india_tz = pytz.timezone('Asia/Kolkata')
    diagnosed_at = datetime.now(india_tz)
    conv = DiagnosisRecord(
        user_id=current_user.id,
        symptoms="CHATBOT: " + user_message,  # Store full query for context
        de_symptoms="CHATBOT: " + ", ".join(extracted_symptoms) if extracted_symptoms else "CHATBOT",
        diagnosed_at=diagnosed_at,
        used_history=1 if last_diagnosis else 0,
        predicted_disease=predicted_disease,
        severity=severity,
        diet_plan=get_disease_details(predicted_disease).get('diet', ''),
        exercise_plan=get_disease_details(predicted_disease).get('exercise', ''),
        medicine=get_disease_details(predicted_disease).get('medicine', ''),
        chatbot_response=bot_reply
    )
    db.session.add(conv)
    db.session.commit()

    # Send doctors in response
    return jsonify({
        "reply": bot_reply,
        "doctors": [
            {
                "id": d.doctor_id,
                "name": d.full_name,
                "specialty": d.specialty,
                "clinic": d.clinic,
                "address": d.address
            } for d in recommended_doctors
        ]
    })


@app.route('/appointments')
@login_required
def my_appointments():
    from models import DoctorVisit, DoctorInfo

    appointments = (
        DoctorVisit.query
        .filter_by(user_id=current_user.id)
        .order_by(DoctorVisit.booked_at.desc())
        .all()
    )

    return render_template(
        'appointments.html',
        appointments=appointments
    )


if __name__ == '__main__':
    app.run(debug=True,
        use_reloader=False)