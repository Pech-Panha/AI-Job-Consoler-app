import sys
import sqlite3
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QScrollArea, QFrame, QMessageBox,
    QTextEdit, QStackedWidget, QRadioButton, QButtonGroup, QSpacerItem, QSizePolicy,
    QListWidget # Added for the left panel career list
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QBrush, QLinearGradient, QPixmap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import json
from collections import defaultdict

# --- Scikit-learn imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Database Setup ---
DATABASE_NAME = 'career_data.db'

def init_db():
    """Initializes the SQLite database and creates the survey_responses table."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            raw_survey_responses TEXT,
            preferred_industry TEXT,
            recommended_career TEXT,
            recommendation_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# --- JOB DETAILS (Incorporating all data from Ai_Jobs.docx) ---
JOB_DETAILS = {
    "Doctor": {
        "description": "Diagnose and treat illnesses, perform check-ups, prescribe medicine, and guide patients to recovery.",
        "salary_range": "$11,000 - $40,000 per year (Annual)",
        "skills": [
            "Clinical Diagnosis",
            "Treatment Planning",
            "Patient Care",
            "Medical Procedures",
            "Communication",
            "Problem-solving"
        ],
        "schools": [
            "University of Health Sciences",
            "International University",
            "Norton University",
            "University of Puthisastra",
            "Cambodian University for Specialties"
        ],
        "companies": [
            "Royal Phnom Penh Hospital",
            "Calmette Hospital",
            "Sunrise Japan Hospital Phnom Penh",
            "SenSok International University Hospital",
            "Raffles Medical Phnom Penh"
        ],
        "image_path": "img/doctor.jpg"
    },
    "Project Manager": {
        "description": "leads and oversees projects from start to finish—planning, coordinating resources, ensuring timelines and budgets are met, managing teams, and communicating with stakeholders.",
        "salary_range": "$5,000 - $19,000 per year (Annual)",
        "skills": [
            "Planning",
            "Budget Management",
            "Team Leadership",
            "Risk Management",
            "Communication"
        ],
        "schools": [
            "CamEd Business School",
            "University of Puthisatra",
            "National Institute of Business",
            "National University of Management",
            "Norton University"
        ],
        "companies": [
            "Vattanac Bank Cambodia",
            "ACLEDA Company",
            "Oddatelier Company"
        ],
        "image_path": "img/Project manager.png"
    },
    "Researcher": {
        "description": "Researchers conduct systematic investigations to establish facts, develop new theories, or revise existing ones. They often work in academic institutions, government agencies, or private companies, designing experiments, collecting and analyzing data, and reporting their findings. Strong analytical skills, attention to detail, and a commitment to scientific integrity are crucial.",
        "salary_range": "$5,000 - $18,000 per year (Annual)",
        "skills": [
            "Research Design",
            "Data Collection",
            "Statistical Analysis",
            "Report Writing",
            "Critical Thinking"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP)",
            "University of Health Sciences (UHS)",
            "Institute of Technology of Cambodia (ITC)",
            "American University of Phnom Penh (AUPP)"
        ],
        "companies": [
            "Universities",
            "Government Labs",
            "Pharmaceutical Companies",
            "R&D Departments",
            "Innovative Research Firms"
        ],
        "image_path": "img/Researcher.png"
    },
    "UX/UI Designer": {
        "description": "focuses on creating intuitive, efficient, and enjoyable user experiences for websites, apps, and software. They research user needs, design interfaces, and test prototypes to ensure products are user-friendly.",
        "salary_range": "$6,000 - $20,000 per year (Annual)",
        "skills": [
            "User Research",
            "Wireframing",
            "Prototyping",
            "Usability Testing",
            "Figma/Sketch/Adobe XD",
            "Communication"
        ],
        "schools": [
            "Limkokwing University of Creative Technology",
            "Royal University of Phnom Penh (RUPP)",
            "Pannasastra University of Cambodia (PUC)",
            "Cambodia Academy of Digital Technology (CADT)"
        ],
        "companies": [
            "Tech Startups",
            "Digital Agencies",
            "E-commerce Companies",
            "Software Development Firms",
            "Banks"
        ],
        "image_path": "img/ux ui.png"
    },
    "Data Scientist": {
        "description": "Data Scientists analyze complex datasets to extract insights and knowledge. They use statistical analysis, machine learning, and programming to build predictive models and inform business decisions. A strong background in mathematics and statistics is beneficial.",
        "salary_range": "$7,000 - $24,000 per year (Annual)",
        "skills": [
            "Programming (Python, R, SQL)",
            "Data Analysis Tools",
            "Machine Learning",
            "Data Visualization",
            "Problem-solving",
            "Critical thinking",
            "Communication"
        ],
        "schools": [
            "American University of Phnom Penh (AUPP)",
            "Institute of Technology of Cambodia (ITC)",
            "Royal University of Phnom Penh (RUPP)",
            "Step IT Academy",
            "Cambodia Academy of Digital Technology (CADT)"
        ],
        "companies": [
            "Banks & Microfinance Company: ABA Bank, Acleda Bank, Wing Bank",
            "Telecom: Smart Axiata, Metfone, Cellcard",
            "Tech Companies / Startups: Codingate, Pathmazing, Slash"
        ],
        "image_path": "img/Data Scientist.png"
    },
    "Software Engineer": {
        "description": "design, develop, and maintain software applications. They apply engineering principles to build robust, scalable, and efficient systems.",
        "salary_range": "$6,000 - $22,000 per year (Annual)",
        "skills": [
            "Programming (Java, Python, C++, JavaScript)",
            "Data Structures & Algorithms",
            "Software Development Life Cycle (SDLC)",
            "Database Management",
            "Problem-solving",
            "Teamwork"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP)",
            "Institute of Technology of Cambodia (ITC)",
            "National University of Management (NUM)",
            "American University of Phnom Penh (AUPP)",
            "SETEC Institute"
        ],
        "companies": [
            "Tech Companies (e.g., Agoda, Pruksa)",
            "Banks & FinTech",
            "Telecoms",
            "Software Outsourcing Firms",
            "E-commerce Platforms"
        ],
        "image_path": "img/Software enginee.png"
    },
    "Fire Fighter": {
        "description": "Firefighters respond to emergencies, extinguish fires, rescue people from dangerous situations, and provide first aid. They also educate the public on fire safety.",
        "salary_range": "$3,000 - $8,000 per year (Annual)",
        "skills": [
            "Emergency Response",
            "First Aid/CPR",
            "Physical Fitness",
            "Teamwork",
            "Stress Management"
        ],
        "schools": [
            "National Police Academy of Cambodia (specific firefighter training programs)",
            "Various provincial training centers"
        ],
        "companies": [
            "Fire and Rescue Department (under Ministry of Interior)",
            "Airport Fire Services",
            "Industrial Fire Brigades (large factories, complexes)"
        ],
        "image_path": "img/firefigher.jpg" 
    },
    "Lawyer": {
        "description": "Lawyers provide legal advice, represent clients in court, and prepare legal documents. They specialize in various fields like criminal law, civil law, or corporate law.",
        "salary_range": "$8,000 - $30,000 per year (Annual)",
        "skills": [
            "Legal Research",
            "Advocacy",
            "Negotiation",
            "Contract Drafting",
            "Communication",
            "Analytical Thinking"
        ],
        "schools": [
            "Royal University of Law and Economics (RULE)",
            "Pannasastra University of Cambodia (PUC)",
            "National University of Management (NUM)",
            "University of Cambodia (UC)"
        ],
        "companies": [
            "Law Firms",
            "Corporate Legal Departments",
            "Government Ministries",
            "NGOs",
            "International Organizations"
        ],
        "image_path": "img/lawyer.jpg"
    },
    "High School Teacher": {
        "description": "High school teachers educate students in various subjects, prepare lesson plans, assess student progress, and foster a positive learning environment.",
        "salary_range": "$3,000 - $10,000 per year (Annual)",
        "skills": [
            "Lesson Planning",
            "Classroom Management",
            "Subject Matter Expertise",
            "Communication",
            "Student Assessment",
            "Adaptability"
        ],
        "schools": [
            "National Institute of Education (NIE)",
            "Royal University of Phnom Penh (RUPP) - Education Dept.",
            "Phnom Penh International University (PPIU) - Education Dept."
        ],
        "companies": [
            "Public High Schools (Ministry of Education, Youth and Sport)",
            "Private International Schools",
            "Community Learning Centers"
        ],
        "image_path": "img/teacher.jpg" 
    },
    "Accountant": {
        "description": "Accountants prepare and examine financial records, ensure financial statements are accurate, and help individuals and businesses manage their finances and comply with tax laws.",
        "salary_range": "$4,000 - $15,000 per year (Annual)",
        "skills": [
            "Financial Reporting",
            "Tax Preparation",
            "Auditing",
            "Bookkeeping",
            "Data Analysis",
            "Attention to Detail"
        ],
        "schools": [
            "CamEd Business School",
            "National University of Management (NUM)",
            "Royal University of Law and Economics (RULE)",
            "University of Cambodia (UC)"
        ],
        "companies": [
            "Accounting Firms",
            "Banks & Financial Institutions",
            "Manufacturing Companies",
            "NGOs",
            "Government Agencies"
        ],
        "image_path": "img/accountant.jpg"
    },
    "Civil Site Engineer": {
        "description": "Civil Site Engineers plan, design, and manage construction projects such as buildings, roads, bridges, and infrastructure, ensuring they are built safely and efficiently.",
        "salary_range": "$5,000 - $18,000 per year (Annual)",
        "skills": [
            "Project Management",
            "Structural Analysis",
            "AutoCAD/Design Software",
            "Site Supervision",
            "Problem-solving",
            "Safety Regulations"
        ],
        "schools": [
            "Institute of Technology of Cambodia (ITC)",
            "National University of Management (NUM)",
            "Norton University",
            "Royal University of Phnom Penh (RUPP) - Engineering Dept."
        ],
        "companies": [
            "Construction Companies",
            "Real Estate Developers",
            "Consulting Engineering Firms",
            "Government Public Works Departments",
            "Infrastructure Development Companies"
        ],
        "image_path": "img/enginee.jpg"
    },
    "Architecture": {
        "description": "Architects design buildings and other physical structures. They blend aesthetics with functionality, considering safety, sustainability, and client needs.",
        "salary_range": "$5,000 - $17,000 per year (Annual)",
        "skills": [
            "Architectural Design",
            "AutoCAD/Revit",
            "Sketching & Rendering",
            "Building Codes",
            "Project Management",
            "Creativity"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP) - Dept. of Architecture",
            "Limkokwing University of Creative Technology",
            "Pannasastra University of Cambodia (PUC) - Architecture"
        ],
        "companies": [
            "Architectural Firms",
            "Construction Companies",
            "Real Estate Development Firms",
            "Interior Design Companies",
            "Government Urban Planning Departments"
        ],
        "image_path": "img/architect.jpg"
    },
    "Artist": {
        "description": "Artists create visual, performing, or literary works. This broad field includes painters, sculptors, musicians, writers, and digital artists, who use their creativity to express ideas and evoke emotions.",
        "salary_range": "$2,000 - $10,000 per year (Annual) - Highly variable",
        "skills": [
            "Creativity",
            "Specific Art Medium (e.g., painting, digital art, music)",
            "Self-promotion",
            "Attention to Detail",
            "Adaptability"
        ],
        "schools": [
            "Royal University of Fine Arts (RUFA)",
            "Limkokwing University of Creative Technology",
            "Phare Ponleu Selpak (Artistic training NGO)"
        ],
        "companies": [
            "Art Galleries",
            "Design Studios",
            "Entertainment Industry",
            "Advertising Agencies",
            "Freelance/Self-employed"
        ],
        "image_path": "img/job1.png"
    },
    "Digital Marketer": {
        "description": "Digital marketers promote products or services online using various digital channels like social media, search engines, email, and websites. They focus on increasing brand awareness, driving traffic, and generating leads.",
        "salary_range": "$2,000 - $9,000 per year (Annual)",
        "skills": [
            "Social Media Marketing",
            "Content Creation",
            "SEO (Search Engine Optimization)",
            "Email Marketing",
            "Google Analytics",
            "Campaign Management"
        ],
        "schools": [
            "National University of Management (NUM) - Marketing",
            "Pannasastra University of Cambodia (PUC) - Marketing",
            "Royal University of Phnom Penh (RUPP) - Media & Communication"
        ],
        "companies": [
            "Digital Marketing Agencies",
            "E-commerce Businesses",
            "Tech Startups",
            "Large Corporations (in-house marketing teams)",
            "NGOs"
        ],
        "image_path": "img/Digital marketer.png"
    },
    "Human Resource (HR)": {
        "description": "manages recruitment, employee relations, training, and company policies to support staff and help the organization run smoothly.",
        "salary_range": "$2,000 - $9,000 per year (Annual)",
        "skills": [
            "Recruitment and interviewing",
            "Knowledge of Cambodian labor law",
            "Payroll and benefits administration",
            "Communication and interpersonal skills",
            "Problem-solving and conflict management"
        ],
        "schools": [
            "Human Resource University",
            "Pannasastra University of Cambodia",
            "Royal University of Phnom Penh",
            "The Knowledge Academy",
            "Cambodian Mekong University"
        ],
        "companies": [
            "private companies",
            "non-profit organizations",
            "government agencies",
            "Consulting Firms",
            "International Organizations"
        ],
        "image_path": "img/HR.jpg"
    }
}


# --- Dummy Data Generation for ML Model Training ---
def generate_dummy_data(num_samples=200):
    """
    Generates a synthetic dataset for training the ML model.
    Maps aggregated survey responses to career outcomes.
    """
    # Define the aggregated features (0-10 scale)
    feature_names = [
        'math_interest', 'science_interest', 'coding_interest', 'design_interest',
        'problem_solving_skill', 'communication_skill', 'creativity_skill', 'leadership_skill'
    ]
    
    # Define possible career outcomes - using all keys from JOB_DETAILS
    career_outcomes = list(JOB_DETAILS.keys())

    data = []
    labels = []

    for _ in range(num_samples):
        # Generate random aggregated scores
        features = {name: np.random.uniform(1, 10) for name in feature_names}
        
        # Simple logic to assign a career based on features (mimicking real patterns)
        # This mapping needs to be expanded to cover all new careers
        career = np.random.choice(career_outcomes) # Default random pick

        # More specific assignments
        if features['coding_interest'] > 7 and features['problem_solving_skill'] > 6:
            career = np.random.choice(["Software Engineer", "Data Scientist"])
        elif features['math_interest'] > 7 and features['science_interest'] > 6:
            career = np.random.choice(["Data Scientist", "Researcher"])
        elif features['design_interest'] > 7 and features['creativity_skill'] > 6:
            career = "UX/UI Designer"
        elif features['leadership_skill'] > 7 and features['communication_skill'] > 6:
            career = np.random.choice(["Project Manager", "Human Resource (HR)"])
        elif features['science_interest'] > 7 and features['problem_solving_skill'] > 6:
            career = np.random.choice(["Doctor", "Researcher"])
        elif features['communication_skill'] > 7 and features['creativity_skill'] > 6:
            career = np.random.choice(["Artist", "Digital Marketer"])
        elif features['math_interest'] > 7 and features['problem_solving_skill'] > 6:
            career = np.random.choice(["Accountant", "Civil Site Engineer", "Architecture"])
        elif features['leadership_skill'] > 7 and features['problem_solving_skill'] > 6:
            career = np.random.choice(["Fire Fighter", "Lawyer"])
        # Add more specific rules as needed for better accuracy

        data.append(list(features.values()))
        labels.append(career)

    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(labels)
    
    return X, y, feature_names, career_outcomes

# --- Machine Learning Model Training ---
def train_career_model():
    """
    Trains a Decision Tree Classifier model for career recommendation.
    """
    X, y, feature_names, career_outcomes = generate_dummy_data()

    # Create a pipeline with scaling and a classifier
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    model_pipeline.fit(X, y)
    print("Machine Learning model trained successfully.")
    return model_pipeline, feature_names, career_outcomes

# --- ML-based Recommendation Logic ---
def get_ml_career_recommendation(ml_model, feature_names, career_outcomes, raw_survey_responses, preferred_industry):
    """
    Uses the trained ML model to get career recommendations.

    Args:
        ml_model (Pipeline): The trained Scikit-learn pipeline.
        feature_names (list): List of feature names used during training.
        career_outcomes (list): List of possible career outcomes.
        raw_survey_responses (dict): A dictionary containing responses to the 20 questions (1-7 scale).
        preferred_industry (str): The user's selected preferred industry.

    Returns:
        tuple: (recommended_career, recommendation_score, top_careers_for_display)
    """
    def map_scale(value):
        return (value - 1) * (10 / 6) # Map 1-7 scale to 0-10

    # Aggregate raw survey responses into interest/skill categories (0-10 scale)
    category_scores = defaultdict(float)
    category_counts = defaultdict(int)

    # Mapping survey questions to aggregated features
    q_to_category = {
        'q1': 'math_interest', 'q2': 'math_interest',
        'q3': 'science_interest', 'q4': 'science_interest',
        'q5': 'coding_interest', 'q6': 'coding_interest', 'q7': 'coding_interest',
        'q8': 'design_interest', 'q9': 'design_interest',
        'q10': 'problem_solving_skill', 'q11': 'problem_solving_skill', 'q12': 'problem_solving_skill',
        'q13': 'communication_skill', 'q14': 'communication_skill', 'q15': 'communication_skill',
        'q16': 'creativity_skill', 'q17': 'creativity_skill',
        'q18': 'leadership_skill', 'q19': 'leadership_skill', 'q20': 'leadership_skill',
    }

    for q_key, response_value in raw_survey_responses.items():
        category = q_to_category.get(q_key)
        if category:
            category_scores[category] += map_scale(response_value)
            category_counts[category] += 1

    aggregated_data = {}
    for feature in feature_names:
        if category_counts[feature] > 0:
            aggregated_data[feature] = category_scores[feature] / category_counts[feature]
        else:
            aggregated_data[feature] = map_scale(4) # Default to neutral (4 on 1-7 scale, mapped to ~5 on 0-10) if no questions contributed

    # Prepare features for the model in the correct order
    input_features = pd.DataFrame([list(aggregated_data.values())], columns=feature_names)

    # Get probability predictions for each career
    probabilities = ml_model.predict_proba(input_features)[0]
    
    # Create a dictionary of career probabilities
    career_probs = {career: prob for career, prob in zip(ml_model.classes_, probabilities)}

    # Apply preferred industry boost (post-prediction)
    industry_boost_factor = 1.2
    career_industry_mapping = {
        "Software Engineer": ["IT", "Technology"],
        "Data Scientist": ["IT", "Research", "Finance"],
        "UX/UI Designer": ["Design", "IT"],
        "Project Manager": ["Management", "General"],
        "Researcher": ["Research", "Science"],
        "Doctor": ["Healthcare"],
        "Fire Fighter": ["Public Service", "General"],
        "Lawyer": ["Legal", "General"],
        "High School Teacher": ["Education", "General"],
        "Accountant": ["Finance", "General"],
        "Civil Site Engineer": ["Engineering", "Construction"],
        "Architecture": ["Design", "Construction"],
        "Artist": ["Arts", "Design"],
        "Digital Marketer": ["Marketing", "IT"],
        "Human Resource (HR)": ["Management", "General"]
    }

    for career, prob in career_probs.items():
        if preferred_industry in career_industry_mapping.get(career, []):
            career_probs[career] = min(1.0, prob * industry_boost_factor) # Cap at 1.0

    # Convert probabilities to scores (e.g., out of 100)
    career_scores = {career: prob * 100 for career, prob in career_probs.items()}

    # Get the top recommended career and its score
    if career_scores:
        recommended_career = max(career_scores, key=career_scores.get)
        recommendation_score = career_scores[recommended_career]
    else:
        recommended_career = "Uncertain"
        recommendation_score = 0.0

    # Get top 3 careers for pie chart display
    sorted_careers = sorted(career_scores.items(), key=lambda item: item[1], reverse=True)
    top_careers_for_display = [(career, score) for career, score in sorted_careers[:3]]

    return recommended_career, recommendation_score, top_careers_for_display


# --- Main Application Window ---
class CareerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ប្រព័ន្ធវិភាគសមត្ថភាព និងផ្តល់យោបល់ការងារ") # Competency Analysis and Career Counseling System
        self.setGeometry(100, 100, 1200, 800) # Increased width

        self.ml_model, self.feature_names, self.career_outcomes = train_career_model()

        self.init_ui()
        init_db()

    def init_ui(self):
        """Initializes the user interface."""
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#e0f2f7"))
        gradient.setColorAt(1.0, QColor("#c2e0f0"))
        palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#333333"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#333333"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#4CAF50"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
        self.setPalette(palette)

        font = QFont("Khmer OS Siemreap", 10)
        self.setFont(font)

        self.stacked_widget = QStackedWidget(self)
        main_layout = QHBoxLayout(self) # Changed to QHBoxLayout for side-by-side layout
        main_layout.addWidget(self.stacked_widget)


        # self.home_page = self.create_home_page() # Removed as per user request
        self.survey_page = self.create_survey_page()
        self.results_page = self.create_results_page()
        self.job_details_page = self.create_job_details_page() # This page will now be used by the left panel
        self.history_page = self.create_history_page()

        # self.stacked_widget.addWidget(self.home_page) # Removed
        self.stacked_widget.addWidget(self.survey_page) # Index 0
        self.stacked_widget.addWidget(self.results_page) # Index 1
        self.stacked_widget.addWidget(self.job_details_page) # Index 2
        self.stacked_widget.addWidget(self.history_page) # Index 3

        # Start on the main career details page with the intro showing by default
        self.stacked_widget.setCurrentIndex(2) # Display job_details_page initially (its new index)


    # Removed create_home_page method as per user request

    def create_survey_page(self):
        """Creates the survey input page widget with improved styling."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(30, 30, 30, 30)

        header_label = QLabel("បំពេញសំណួរស្ទង់មតិ (២០ សំណួរ)")
        header_label.setFont(QFont("Khmer OS Muol Light", 18))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #2c3e50; margin-bottom: 25px;")
        main_layout.addWidget(header_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: 1px solid #d0d0d0; border-radius: 10px; background-color: rgba(255,255,255,0.8); }")
        scroll_content = QWidget()
        self.survey_layout = QVBoxLayout(scroll_content)
        self.survey_layout.setContentsMargins(25, 25, 25, 25)
        self.survey_layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        student_info_prompt_label = QLabel("សូមបំពេញព័តមានរបស់សិស្សជាមុនសិនៈ")
        student_info_prompt_label.setFont(QFont("Khmer OS Siemreap", 12)) # You can adjust '12' to your desired size
        self.survey_layout.addWidget(student_info_prompt_label)

        student_name_label = QLabel("ឈ្មោះនិស្សិត*:")
        student_name_label.setFont(QFont("Khmer OS Siemreap", 12)) # Increased font size
        self.survey_layout.addWidget(student_name_label)
        self.student_name_input = QLineEdit()
        self.student_name_input.setPlaceholderText("បញ្ចូលឈ្មោះរបស់អ្នក")
        self.student_name_input.setFont(QFont("Khmer OS Siemreap", 11))
        self.student_name_input.setStyleSheet(
            "QLineEdit { "
            "padding: 10px; border: 1px solid #a0a0a0; border-radius: 8px; "
            "background-color: #f8f8f8; "
            "}"
            "QLineEdit:focus { border: 2px solid #007bff; background-color: #ffffff; }"
        )
        self.survey_layout.addWidget(self.student_name_input)
        self.survey_layout.addSpacing(25)

        industry_label = QLabel("ឧស្សាហកម្មដែលពេញចិត្ត:")
        industry_label.setFont(QFont("Khmer OS Siemreap", 12)) # Increased font size
        self.survey_layout.addWidget(industry_label)
        self.survey_industry_combo = QComboBox()
        # Updated industry options to cover all new jobs better
        self.survey_industry_combo.addItems([
            "IT", "Design", "Management", "Research", "Finance",
            "Education", "Healthcare", "Public Service", "Legal",
            "Construction", "Engineering", "Arts", "Marketing", "General"
        ])
        self.survey_industry_combo.setFont(QFont("Khmer OS Siemreap", 11))
        self.survey_industry_combo.setStyleSheet(
            "QComboBox { "
            "padding: 10px; border: 1px solid #a0a0a0; border-radius: 8px; "
            "background-color: #f8f8f8; "
            "selection-background-color: #007bff; "
            "}"
            "QComboBox::drop-down { border: none; }"
        )
        self.survey_layout.addWidget(self.survey_industry_combo)
        self.survey_layout.addSpacing(30)


        self.questions = [
            "I enjoy solving complex mathematical problems.",
            "I find working with numbers and data engaging.",
            "I am curious about how the natural world and scientific principles work.",
            "I enjoy learning about new scientific discoveries and theories.",
            "I enjoy writing code and developing software applications.",
            "I like to build and troubleshoot computer systems.",
            "I am interested in artificial intelligence and machine learning.",
            "I enjoy creating visual designs and artistic layouts.",
            "I pay attention to user experience and interface aesthetics.",
            "I enjoy breaking down complex problems into smaller, manageable parts.",
            "I like to find innovative solutions to challenges.",
            "I am good at analytical thinking and logical reasoning.",
            "I am comfortable presenting ideas and information to groups.",
            "I enjoy collaborating with others to achieve a common goal.",
            "I can clearly explain complex topics to different audiences.",
            "I enjoy generating new and original ideas.",
            "I am comfortable thinking outside the box and experimenting.",
            "I like to take charge and guide a team towards a goal.",
            "I am good at organizing tasks and managing resources.",
            "I enjoy motivating others and resolving conflicts."
        ]

        self.question_button_groups = {}
        for i, q_text in enumerate(self.questions):
            q_num = i + 1
            question_label = QLabel(f"សំណួរ {q_num}: {q_text}")
            question_label.setFont(QFont("Khmer OS Siemreap", 11, QFont.Weight.Bold))
            question_label.setStyleSheet("color: #333333;")
            self.survey_layout.addWidget(question_label)

            h_layout = QHBoxLayout()
            h_layout.setSpacing(10)

            button_group = QButtonGroup(self)
            self.question_button_groups[f'q{q_num}'] = button_group

            disagree_label = QLabel("Disagree")
            disagree_label.setFont(QFont("Khmer OS Siemreap", 10, QFont.Weight.Bold))
            disagree_label.setStyleSheet("color: #dc3545;")
            h_layout.addWidget(disagree_label)

            for val in range(1, 8):
                radio_button = QRadioButton()
                radio_button.setProperty("value", val)
                radio_button.setFixedSize(30, 30) # Make radio buttons larger
                radio_button.setStyleSheet(
                    "QRadioButton::indicator { "
                    "width: 25px; height: 25px; border-radius: 12px; "
                    "border: 2px solid #a0a0a0; "
                    "background-color: #f0f0f0; "
                    "}"
                    "QRadioButton::indicator:unchecked:hover { background-color: #e0e0e0; }"
                    "QRadioButton::indicator:checked { "
                    "background-color: #4CAF50; "
                    "border: 2px solid #4CAF50; "
                    "}"
                )
                if val == 4: # Default to neutral (middle)
                    radio_button.setChecked(True)
                h_layout.addWidget(radio_button, alignment=Qt.AlignmentFlag.AlignCenter)
                button_group.addButton(radio_button, val)

            agree_label = QLabel("Agree")
            agree_label.setFont(QFont("Khmer OS Siemreap", 10, QFont.Weight.Bold))
            agree_label.setStyleSheet("color: #28a745;")
            h_layout.addWidget(agree_label)
            
            h_layout.addStretch(1) # Pushes radio buttons to the left
            self.survey_layout.addLayout(h_layout)
            self.survey_layout.addSpacing(15)

        self.survey_layout.addSpacing(30)

        submit_button = QPushButton("បំពេញការស្ទង់មតិ")
        submit_button.setFont(QFont("Khmer OS Siemreap", 14, QFont.Weight.Bold))
        submit_button.setFixedSize(250, 55)
        submit_button.setStyleSheet(
            "QPushButton { "
            "background-color: #28a745; color: white; border-radius: 27px; "
            "border: none; padding: 10px 20px; "
            "}"
            "QPushButton:hover { "
            "background-color: #218838; "
            "}"
        )
        submit_button.clicked.connect(self.submit_survey)
        self.survey_layout.addWidget(submit_button, alignment=Qt.AlignmentFlag.AlignCenter)

        back_button = QPushButton("ត្រឡប់ទៅការងារ")
        back_button.setFont(QFont("Khmer OS Siemreap", 11))
        back_button.setFixedSize(180, 60)
        back_button.setStyleSheet(
            "QPushButton { "
            "background-color: #6c757d; color: white; border-radius: 20px; "
            "border: none; padding: 8px 15px; margin-top: 15px; "
            "}"
            "QPushButton:hover { "
            "background-color: #5a6268; "
            "}"
        )
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2)) # Back to job details page (new index)
        self.survey_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.survey_layout.addStretch(1) # Push content to the top
        return widget

    def submit_survey(self):
        """Collects survey responses, gets ML recommendation, saves to DB, and shows results."""
        student_name = self.student_name_input.text().strip()
        if not student_name:
            QMessageBox.warning(self, "Missing Information", "សូមបញ្ចូលឈ្មោះនិស្សិត។") # Please enter student name.
            return

        raw_responses = {}
        all_questions_answered = True
        for q_key, btn_group in self.question_button_groups.items():
            checked_button = btn_group.checkedButton()
            if checked_button:
                raw_responses[q_key] = checked_button.property("value")
            else:
                all_questions_answered = False
                break
        
        if not all_questions_answered:
            QMessageBox.warning(self, "Missing Responses", "សូមឆ្លើយគ្រប់សំណួរទាំងអស់។") # Please answer all questions.
            return

        preferred_industry = self.survey_industry_combo.currentText()

        # Get ML recommendation
        recommended_career, recommendation_score, top_careers_for_display = \
            get_ml_career_recommendation(
                self.ml_model, self.feature_names, self.career_outcomes,
                raw_responses, preferred_industry
            )

        # Save to database
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO survey_responses (student_name, raw_survey_responses, preferred_industry, recommended_career, recommendation_score) VALUES (?, ?, ?, ?, ?)",
                (student_name, json.dumps(raw_responses), preferred_industry, recommended_career, recommendation_score)
            )
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Submission Successful", "ការស្ទង់មតិត្រូវបានដាក់ស្នើដោយជោគជ័យ!")
            self.show_results_page(student_name, recommended_career, top_careers_for_display[0][1], top_careers_for_display) # Pass the score of the top career for display
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"មានបញ្ហាជាមួយមូលដ្ឋានទិន្នន័យ: {e}")


    def create_results_page(self):
        """Creates the results display page."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        header_label = QLabel("លទ្ធផលនៃការណែនាំអាជីព")
        header_label.setFont(QFont("Khmer OS Muol Light", 22))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #004d40; margin-bottom: 20px;")
        layout.addWidget(header_label)

        self.student_name_result_label = QLabel("")
        self.student_name_result_label.setFont(QFont("Khmer OS Siemreap", 14))
        self.student_name_result_label.setStyleSheet("color: #333333;")
        layout.addWidget(self.student_name_result_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.recommended_career_label = QLabel("")
        self.recommended_career_label.setFont(QFont("Khmer OS Muol Light", 20))
        self.recommended_career_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recommended_career_label.setStyleSheet("color: #2e7d32; margin-top: 10px; margin-bottom: 10px;")
        layout.addWidget(self.recommended_career_label)

        self.recommendation_score_label = QLabel("")
        self.recommendation_score_label.setFont(QFont("Khmer OS Siemreap", 16))
        self.recommendation_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recommendation_score_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.recommendation_score_label)

        # Matplotlib figure for pie chart
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(QSize(400, 400)) # Set a minimum size for the chart
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        details_button = QPushButton("មើលព័ត៌មានលម្អិតការងារ")
        details_button.setFont(QFont("Khmer OS Siemreap", 12))
        details_button.setFixedSize(220, 50)
        details_button.setStyleSheet(
            "QPushButton { "
            "background-color: #007bff; color: white; border-radius: 25px; "
            "border: none; padding: 10px 20px; margin-top: 20px; "
            "}"
            "QPushButton:hover { "
            "background-color: #0069d9; "
            "}"
        )
        details_button.clicked.connect(self.show_recommended_job_details)
        layout.addWidget(details_button, alignment=Qt.AlignmentFlag.AlignCenter)

        new_survey_button = QPushButton("ធ្វើការស្ទង់មតិថ្មី")
        new_survey_button.setFont(QFont("Khmer OS Siemreap", 12))
        new_survey_button.setFixedSize(200, 50)
        new_survey_button.setStyleSheet(
            "QPushButton { "
            "background-color: #6c757d; color: white; border-radius: 25px; "
            "border: none; padding: 10px 20px; margin-top: 10px; "
            "}"
            "QPushButton:hover { "
            "background-color: #5a6268; "
            "}"
        )
        new_survey_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0)) # Back to survey (new index)
        layout.addWidget(new_survey_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1) # Push content to the top
        return widget

    def show_results_page(self, student_name, recommended_career, recommendation_score, top_careers_for_display):
        """Displays the results page with the recommendation."""
        self.student_name_result_label.setText(f"ឈ្មោះនិស្សិត: {student_name}")
        self.recommended_career_label.setText(f"អាជីពដែលបានណែនាំ: {recommended_career}")
        self.recommendation_score_label.setText(f"ពិន្ទុភាពស័ក្តិសម: {recommendation_score:.2f}%")
        self.current_recommended_career = recommended_career # Store for details button

        # Clear previous plot
        self.ax.clear()

        # Prepare data for the pie chart
        labels = [f"{career} ({score:.1f}%)" for career, score in top_careers_for_display]
        sizes = [score for career, score in top_careers_for_display]
        colors = ['#4CAF50', '#FFC107', '#2196F3'] # Green, Yellow, Blue for top 3
        explode = [0.1 if i == 0 else 0 for i in range(len(top_careers_for_display))] # Explode the largest slice

        self.ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', # Only show percentage if > 0
                    shadow=True, startangle=140, textprops={'fontsize': 10, 'color': 'black', 'fontname': 'Khmer OS Siemreap'})
        self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.ax.set_title("ការណែនាំអាជីពកំពូលទាំង ៣", fontsize=14, fontname='Khmer OS Muol Light')

        self.canvas.draw()
        self.stacked_widget.setCurrentIndex(1) # Show results page (new index)

    def show_recommended_job_details(self):
        """Switches to the job details page for the currently recommended career."""
        if hasattr(self, 'current_recommended_career') and self.current_recommended_career:
            self.display_job_details(self.current_recommended_career)
        else:
            QMessageBox.warning(self, "No Career Selected", "សូមបំពេញការស្ទង់មតិជាមុនសិន ដើម្បីមើលព័ត៌មានលម្អិតការងារដែលបានណែនាំ។")
        
    def create_history_page(self):
        """Creates the survey history page."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)

        header_label = QLabel("ប្រវត្តិការស្ទង់មតិ")
        header_label.setFont(QFont("Khmer OS Muol Light", 20))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #2c3e50; margin-bottom: 25px;")
        layout.addWidget(header_label)

        self.history_list_widget = QListWidget()
        self.history_list_widget.setFont(QFont("Khmer OS Siemreap", 11))
        self.history_list_widget.setStyleSheet(
            "QListWidget { "
            "border: 1px solid #d0d0d0; border-radius: 10px; background-color: rgba(255,255,255,0.8); "
            "padding: 10px; "
            "}"
            "QListWidget::item { padding: 8px; margin-bottom: 5px; border-bottom: 1px solid #e0e0e0; }"
            "QListWidget::item:selected { background-color: #e6f7ff; color: #007bff; }"
        )
        self.history_list_widget.itemClicked.connect(self.display_history_details)
        layout.addWidget(self.history_list_widget)

        self.history_details_text = QTextEdit()
        self.history_details_text.setReadOnly(True)
        self.history_details_text.setFont(QFont("Khmer OS Siemreap", 10))
        self.history_details_text.setStyleSheet(
            "QTextEdit { "
            "border: 1px solid #d0d0d0; border-radius: 8px; background-color: #f8f8f8; "
            "padding: 15px; "
            "}"
        )
        self.history_details_text.setMinimumHeight(150)
        layout.addWidget(self.history_details_text)

        back_button = QPushButton("ត្រឡប់ទៅការងារ")
        back_button.setFont(QFont("Khmer OS Siemreap", 11))
        back_button.setFixedSize(180, 50)
        back_button.setStyleSheet(
            "QPushButton { "
            "background-color: #6c757d; color: white; border-radius: 25px; "
            "border: none; padding: 10px 20px; margin-top: 20px; "
            "}"
            "QPushButton:hover { "
            "background-color: #5a6268; "
            "}"
        )
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2)) # Back to job details page (new index)
        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        return widget

    def show_history_page(self):
        """Fetches and displays survey history."""
        self.history_list_widget.clear()
        self.history_details_text.clear()
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT student_name, recommended_career, recommendation_score, timestamp, raw_survey_responses, preferred_industry FROM survey_responses ORDER BY timestamp DESC")
            records = cursor.fetchall()
            conn.close()

            self.history_data = [] # Store full data for details
            for i, record in enumerate(records):
                student_name, career, score, timestamp, raw_responses_json, preferred_industry = record
                display_text = f"{i+1}. ឈ្មោះ: {student_name} | អាជីពណែនាំ: {career} ({score:.2f}%) | ថ្ងៃទី: {timestamp}"
                self.history_list_widget.addItem(display_text)
                self.history_data.append({
                    "student_name": student_name,
                    "recommended_career": career,
                    "recommendation_score": score,
                    "timestamp": timestamp,
                    "raw_survey_responses": json.loads(raw_responses_json),
                    "preferred_industry": preferred_industry
                })
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"មានបញ្ហាជាមួយមូលដ្ឋានទិន្នន័យនៅពេលទាញយកប្រវត្តិ: {e}")
        
        self.stacked_widget.setCurrentIndex(3) # Show history page (new index)

    def display_history_details(self, item):
        """Displays detailed information for a selected history item."""
        index = self.history_list_widget.row(item)
        data = self.history_data[index]

        details = f"<h3>ព័ត៌មានលម្អិតនៃការស្ទង់មតិ</h3>" \
                  f"<p><b>ឈ្មោះនិស្សិត:</b> {data['student_name']}</p>" \
                  f"<p><b>ឧស្សាហកម្មដែលពេញចិត្ត:</b> {data['preferred_industry']}</p>" \
                  f"<p><b>អាជីពដែលបានណែនាំ:</b> {data['recommended_career']}</p>" \
                  f"<p><b>ពិន្ទុភាពស័ក្តិសម:</b> {data['recommendation_score']:.2f}%</p>" \
                  f"<p><b>កាលបរិច្ឆេទ:</b> {data['timestamp']}</p>" \
                  f"<br><h4>ចម្លើយស្ទង់មតិ:</h4>"
        
        # Convert raw responses back to aggregated form for display (optional, but useful)
        # This requires re-applying the aggregation logic or saving aggregated scores directly
        # For simplicity, we'll just show raw responses if needed, or stick to summary.
        # Here, let's just show the scores mapped back for clarity.
        
        # Dummy aggregation for display based on raw responses
        feature_names_map = {
            'q1': 'math_interest', 'q2': 'math_interest',
            'q3': 'science_interest', 'q4': 'science_interest',
            'q5': 'coding_interest', 'q6': 'coding_interest', 'q7': 'coding_interest',
            'q8': 'design_interest', 'q9': 'design_interest',
            'q10': 'problem_solving_skill', 'q11': 'problem_solving_skill', 'q12': 'problem_solving_skill',
            'q13': 'communication_skill', 'q14': 'communication_skill', 'q15': 'communication_skill',
            'q16': 'creativity_skill', 'q17': 'creativity_skill',
            'q18': 'leadership_skill', 'q19': 'leadership_skill', 'q20': 'leadership_skill',
        }
        
        # Reverse map for display (question number to original text)
        question_texts = {f"q{i+1}": text for i, text in enumerate(self.questions)}

        response_details = "<p><b>ចម្លើយលម្អិត (1=មិនយល់ស្របខ្លាំង, 7=យល់ស្របខ្លាំង):</b></p><ul>"
        for q_key in sorted(data['raw_survey_responses'].keys(), key=lambda x: int(x[1:])):
            response_details += f"<li>{question_texts.get(q_key, q_key)}: {data['raw_survey_responses'][q_key]}</li>"
        response_details += "</ul>"
        
        self.history_details_text.setHtml(details + response_details)

    def create_job_details_page(self):
        """
        Creates the combined left panel (career list) and the right panel
        (job details display or intro page).
        """
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0) # No margins for the main layout

        # --- Left Panel: Career List and Navigation ---
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet(
            "QFrame { "
            "background-color: #2c3e50; border-right: 1px solid #34495e; "
            "border-top-left-radius: 15px; border-bottom-left-radius: 15px; "
            "}"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 20, 15, 20)
        left_layout.setSpacing(10)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # App Title
        app_title_label = QLabel("Career Compass")
        app_title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        app_title_label.setStyleSheet("color: #ecf0f1; margin-bottom: 20px;")
        left_layout.addWidget(app_title_label)

        # Survey Button (Replaces Home button in the left panel's general navigation)
        survey_button = QPushButton("បំពេញការស្ទង់មតិ")
        survey_button.setFont(QFont("Khmer OS Siemreap", 12))
        survey_button.setStyleSheet(
            "QPushButton { "
            "background-color: #34495e; color: #ecf0f1; border: none; padding: 12px; border-radius: 8px; text-align: left;"
            "}"
            "QPushButton:hover { background-color: #3b536b; }"
            "QPushButton:pressed { background-color: #2c3e50; }"
        )
        survey_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0)) # Link to survey page (new index)
        left_layout.addWidget(survey_button)

        # History Button
        history_button = QPushButton("មើលប្រវត្តិ")
        history_button.setFont(QFont("Khmer OS Siemreap", 12))
        history_button.setStyleSheet(
            "QPushButton { "
            "background-color: #34495e; color: #ecf0f1; border: none; padding: 12px; border-radius: 8px; text-align: left;"
            "}"
            "QPushButton:hover { background-color: #3b536b; }"
            "QPushButton:pressed { background-color: #2c3e50; }"
        )
        history_button.clicked.connect(self.show_history_page)
        left_layout.addWidget(history_button)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Career List Label
        career_list_label = QLabel("ប្រភេទការងារ:")
        career_list_label.setFont(QFont("Khmer OS Muol Light", 14))
        career_list_label.setStyleSheet("color: #ecf0f1; margin-top: 15px; margin-bottom: 5px;")
        left_layout.addWidget(career_list_label)

        # Career List
        self.job_list_widget = QListWidget()
        self.job_list_widget.setFont(QFont("Khmer OS Siemreap", 10))
        self.job_list_widget.setStyleSheet(
            "QListWidget { "
            "background-color: #3b536b; border: 1px solid #4a6782; border-radius: 8px; color: #ecf0f1; "
            "padding: 5px; "
            "}"
            "QListWidget::item { padding: 8px; border-bottom: 1px solid #4a6782; }"
            "QListWidget::item:hover { background-color: #5d7a96; }"
            "QListWidget::item:selected { background-color: #1abc9c; color: white; border-radius: 5px; }"
        )
        for job_name in sorted(JOB_DETAILS.keys()):
            self.job_list_widget.addItem(job_name)
        self.job_list_widget.itemClicked.connect(lambda item: self.display_job_details(item.text()))
        left_layout.addWidget(self.job_list_widget)

        main_layout.addWidget(left_panel)

        # --- Right Panel: Job Details or Intro Page ---
        right_panel = QFrame()
        right_panel.setStyleSheet(
            "QFrame { "
            "background-color: #ecf0f1; border-top-right-radius: 15px; border-bottom-right-radius: 15px; "
            "}"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(30, 30, 30, 30)

        # QStackedWidget for intro vs. job details
        self.right_panel_stacked_widget = QStackedWidget()
        
        # Add intro page
        intro_page = self.create_intro_page_for_job_details()
        self.right_panel_stacked_widget.addWidget(intro_page) # Index 0

        # Create a container for the actual job details content
        self.job_details_content_widget = QWidget()
        job_details_content_layout = QVBoxLayout(self.job_details_content_widget)
        job_details_content_layout.setContentsMargins(0, 0, 0, 0)
        job_details_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)


        # Image Label
        self.job_image_label = QLabel()
        self.job_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.job_image_label.setFixedSize(200, 200) # Ensure consistent size
        self.job_image_label.setStyleSheet("border: 1px solid #ddd; border-radius: 10px; background-color: #fff; padding: 5px;")
        job_details_content_layout.addWidget(self.job_image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Job Title
        self.job_title_label = QLabel("ជ្រើសរើសប្រភេទការងារពីបញ្ជី")
        self.job_title_label.setFont(QFont("Khmer OS Muol Light", 20))
        self.job_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.job_title_label.setStyleSheet("color: #2c3e50; margin-top: 15px; margin-bottom: 10px;")
        job_details_content_layout.addWidget(self.job_title_label)

        # Scrollable Area for Details
        details_scroll_area = QScrollArea()
        details_scroll_area.setWidgetResizable(True)
        details_scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        details_container = QWidget()
        self.details_layout = QVBoxLayout(details_container)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(10)
        self.details_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Description
        self.job_description_label = QLabel("")
        self.job_description_label.setFont(QFont("Khmer OS Siemreap", 11))
        self.job_description_label.setWordWrap(True)
        self.job_description_label.setStyleSheet("color: #34495e; line-height: 1.5;")
        self.details_layout.addWidget(self.job_description_label)

        # Salary
        self.salary_label = QLabel("")
        self.salary_label.setFont(QFont("Khmer OS Siemreap", 11, QFont.Weight.Bold))
        self.salary_label.setStyleSheet("color: #2980b9; margin-top: 10px;")
        self.details_layout.addWidget(self.salary_label)

        # Skills
        self.skills_label = QLabel("")
        self.skills_label.setFont(QFont("Khmer OS Siemreap", 11))
        self.skills_label.setWordWrap(True)
        self.skills_label.setStyleSheet("color: #34495e; margin-top: 5px;")
        self.details_layout.addWidget(self.skills_label)

        # Schools
        self.schools_label = QLabel("")
        self.schools_label.setFont(QFont("Khmer OS Siemreap", 11))
        self.schools_label.setWordWrap(True)
        self.schools_label.setStyleSheet("color: #34495e; margin-top: 5px;")
        self.details_layout.addWidget(self.schools_label)

        # Companies
        self.companies_label = QLabel("")
        self.companies_label.setFont(QFont("Khmer OS Siemreap", 11))
        self.companies_label.setWordWrap(True)
        self.companies_label.setStyleSheet("color: #34495e; margin-top: 5px;")
        self.details_layout.addWidget(self.companies_label)
        
        self.details_layout.addStretch(1) # Pushes content to the top

        details_scroll_area.setWidget(details_container)
        job_details_content_layout.addWidget(details_scroll_area)
        
        self.right_panel_stacked_widget.addWidget(self.job_details_content_widget) # Index 1 for actual details
        
        right_layout.addWidget(self.right_panel_stacked_widget)
        main_layout.addWidget(right_panel)

        self.right_panel_stacked_widget.setCurrentIndex(0) # Show intro page by default

        return widget

    def create_intro_page_for_job_details(self):
        """
        Creates the introductory page for the right panel when no job is selected.
        """
        intro_widget = QWidget()
        intro_layout = QVBoxLayout(intro_widget)
        intro_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro_layout.setSpacing(20)
        intro_layout.setContentsMargins(50, 50, 50, 50)

        welcome_label = QLabel("សូមស្វាគមន៍មកកាន់កម្មវិធី Career Compass!")
        welcome_label.setFont(QFont("Khmer OS Muol Light", 22))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        intro_layout.addWidget(welcome_label)

        description_label = QLabel(
            "កម្មវិធីនេះត្រូវបានរចនាឡើងដើម្បីជួយអ្នកស្វែងរកផ្លូវអាជីពដែលស័ក្តិសមបំផុតសម្រាប់ចំណាប់អារម្មណ៍ និងសមត្ថភាពរបស់អ្នក។ "
            "យើងផ្តល់ជូននូវការវិភាគស៊ីជម្រៅ និងព័ត៌មានលម្អិតអំពីអាជីពនានា ដើម្បីជាជំនួយក្នុងការសម្រេចចិត្តរបស់អ្នក។"
        )
        description_label.setFont(QFont("Khmer OS Siemreap", 11))
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setStyleSheet("color: #34495e; line-height: 1.6;")
        intro_layout.addWidget(description_label)

        # Add an image
        image_label = QLabel()
        pixmap = QPixmap("img/allbots.png") # Using one of the uploaded images
        if not pixmap.isNull():
            image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intro_layout.addWidget(image_label)

        instruction_label = QLabel(
            "សូមចុចប៊ូតុង 'បំពេញការស្ទង់មតិ' នៅផ្នែកខាងឆ្វេង ដើម្បីចាប់ផ្តើមវិភាគសមត្ថភាពរបស់អ្នក និងទទួលបានការណែនាំអាជីពផ្ទាល់ខ្លួន។"
        )
        instruction_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction_label.setStyleSheet("color: #16a085; margin-top: 25px;")
        intro_layout.addWidget(instruction_label)

        intro_layout.addStretch(1) # Push content to the top
        return intro_widget

    def display_job_details(self, job_name):
        """Displays details for the selected job on the right panel."""
        job_info = JOB_DETAILS.get(job_name)
        if job_info:
            self.right_panel_stacked_widget.setCurrentIndex(1) # Show job details content
            self.job_title_label.setText(job_name)
            self.job_description_label.setText(f"<b>ការពិពណ៌នា:</b> {job_info['description']}")
            self.salary_label.setText(f"<b>ជួរប្រាក់ខែ:</b> {job_info['salary_range']}")
            self.skills_label.setText("<b>ជំនាញដែលត្រូវការ:</b> " + ", ".join(job_info['skills']))
            self.schools_label.setText("<b>សាលាដែលបានណែនាំ:</b> " + ", ".join(job_info['schools']))
            self.companies_label.setText("<b>ក្រុមហ៊ុនដែលពាក់ព័ន្ធ:</b> " + ", ".join(job_info['companies']))

            # Load image
            pixmap = QPixmap(job_info['image_path'])
            if not pixmap.isNull():
                self.job_image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                self.job_image_label.clear()
                self.job_image_label.setText("Image Not Found")
                self.job_image_label.setStyleSheet("color: red;")
        else:
            self.right_panel_stacked_widget.setCurrentIndex(0) # Go back to intro if job_name is invalid or None

if __name__ == '__main__':
    # Ensure the img directory exists and contains the images
    # This is a placeholder for actual image loading and path management
    # In a real app, you'd ensure these paths are correct or bundle resources
    import os
    if not os.path.exists("img"):
        os.makedirs("img")
    # You would typically place your images like doctor.jpg, allbots.png, etc. inside the 'img' directory

    app = QApplication(sys.argv)
    window = CareerApp()
    window.showMaximized() # Show maximized for better view of both panels
    sys.exit(app.exec())