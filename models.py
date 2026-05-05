from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.String(10), primary_key=True)  # Changed from Integer to String(10)
    full_name = db.Column(db.String(100))
    profile_photo = db.Column(db.String(255))
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile = db.Column(db.String(20))
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile = db.relationship('UserProfile', backref='user', uselist=False)
    diagnoses = db.relationship('DiagnosisRecord', backref='user', lazy=True)
    emergency_contact_name = db.Column(db.String(100))
    emergency_relationship = db.Column(db.String(50))
    emergency_contact_mobile = db.Column(db.String(20))
    doctor_visits = db.relationship('DoctorVisit', backref='user', lazy=True)

class UserProfile(db.Model):
    __tablename__ = 'user_profile'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(10), db.ForeignKey('users.id'))  # Changed to String(10)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    blood_group = db.Column(db.String(10))
    chronic_conditions = db.Column(db.Text)
    allergies = db.Column(db.Text)
    notes = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    current_medication = db.Column(db.String(255))

class DiagnosisRecord(db.Model):
    __tablename__ = 'diagnosis_record'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(10), db.ForeignKey('users.id'))  # Changed to String(10)
    symptoms = db.Column(db.Text)
    de_symptoms = db.Column(db.Text)  # This is correct for storing processed symptoms
    diagnosed_at = db.Column(db.DateTime)
    used_history = db.Column(db.Boolean)
    predicted_disease = db.Column(db.String(100))
    severity = db.Column(db.String(50))
    diet_plan = db.Column(db.Text)
    exercise_plan = db.Column(db.Text)
    medicine = db.Column(db.Text)  # This is the correct column for suggested medicine
    chatbot_response = db.Column(db.Text) # To store raw chatbot responses

class DoctorInfo(db.Model):
    __tablename__ = 'doctor_info'
    doctor_id = db.Column(db.String(10), primary_key=True)  # Changed from Integer to String(10)
    full_name = db.Column(db.String(100))      # Should match User.full_name
    clinic = db.Column(db.String(255))
    address = db.Column(db.String(255))
    specialty = db.Column(db.String(100))
    clinic_photo = db.Column(db.String(255))
    
class DoctorVisit(db.Model):
    __tablename__ = 'doctor_visit'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    appointment_datetime = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')
    disease = db.Column(db.String(100))
    blood_report = db.Column(db.Text)
    suggested_medicine = db.Column(db.Text)
    notes = db.Column(db.Text)
    booked_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.String(10), db.ForeignKey('users.id'))
    doctor_id = db.Column(db.String(10), db.ForeignKey('doctor_info.doctor_id'))  # Ensure this is String
    doctor = db.relationship('DoctorInfo', backref='visits', lazy=True, primaryjoin="DoctorVisit.doctor_id==DoctorInfo.doctor_id")