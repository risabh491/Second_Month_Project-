from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load data
sym_des = pd.read_csv("symptoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Load model
svc = pickle.load(open('svc.pkl', 'rb'))

# Load training data to match feature order
training_data = pd.read_csv("Training.csv")
all_symptoms = training_data.columns[:-1].tolist()  # Get symptom columns (exclude 'Disease')

# Dynamically generate symptoms dictionary
symptoms_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Disease list as per training label encoding
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
    30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
    5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Optional: mapping user-friendly inputs to correct symptom keys
user_friendly_symptoms_map = {
    'fever': 'high_fever',
    'tired': 'fatigue',
    'cold': 'cough',
    'rash': 'skin_rash',
    'nausea': 'nausea',
    'vomit': 'vomiting',
    'pain': 'abdominal_pain',
    'burning urine': 'burning_micturition',
    # Add more as needed
}

# Helper function to fetch details
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'])
    pre = [col for col in precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values]
    med = [med for med in medications[medications['Disease'] == dis]['Medication'].values]
    die = [die for die in diets[diets['Disease'] == dis]['Diet'].values]
    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout

# Predict function
def get_predicted_value(patient_symptoms):
    if not patient_symptoms:
        raise ValueError("No symptoms provided")

    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        item = item.lower()
        item = user_friendly_symptoms_map.get(item, item)
        if item not in symptoms_dict:
            raise KeyError(item)
        input_vector[symptoms_dict[item]] = 1

    prediction = svc.predict([input_vector])[0]
    return diseases_list[prediction]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip("[]' ").lower() for s in symptoms.split(',')]

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions_list, medications, rec_diet, workout_list = helper(predicted_disease)
            my_precautions = precautions_list[0] if precautions_list else []
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout_list)
        except KeyError as e:
            return render_template('index.html', message=f"Invalid symptom entered: {e.args[0]}")
        except ValueError as e:
            return render_template('index.html', message=str(e))

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

