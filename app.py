from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models import db, User, Prediction

import numpy as np
import cv2
import joblib
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.feature import local_binary_pattern

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE ================= #
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ================= LOGIN ================= #
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= MODELS ================= #
feature_extractor1 = load_model("feature_extractor_model1 (2).h5")
svm_model1        = joblib.load("svm_model1 (2).pkl")
label_encoder1    = joblib.load("label_encoder_model1 (2).pkl")

feature_extractor2 = load_model("feature_extractor_model2 (1).h5")
svm_model2         = joblib.load("svm_model2 (1).pkl")
label_encoder2     = joblib.load("label_encoder_model2 (1).pkl")

# ================= DISEASE INFO ================= #
DISEASE_INFO = {
    "Stem Canker": {
        "cause": "Caused by the fungus Neoscytalidium dimidiatum, which thrives in warm, humid conditions. Enters through wounds or natural openings in the stem, causing dark sunken lesions.",
        "organic": "Remove and destroy infected stem sections. Apply diluted neem oil (5 ml per litre) weekly. Improve air circulation by pruning overcrowded stems. Avoid overhead irrigation.",
        "chemical": "Apply copper oxychloride (0.3%) or Mancozeb (0.2%) fungicide every 10–14 days. In severe cases, use Carbendazim (0.1%) as a drench around the base."
    },
    "Anthracnose": {
        "cause": "Caused by the fungus Colletotrichum gloeosporioides. Spreads rapidly in wet and humid weather through water splashes, causing orange-brown sunken spots on stems and fruit.",
        "organic": "Spray baking soda (1 tsp) and neem oil (5 ml) per litre of water. Remove infected material immediately. Apply Trichoderma-based bio-fungicide as a preventive treatment.",
        "chemical": "Spray Mancozeb (0.25%) or Azoxystrobin (0.1%) every 7–10 days during wet seasons. Alternate chemicals to prevent resistance. Ensure thorough coverage of all stem surfaces."
    },
    "Cactus Virus X": {
        "cause": "A viral disease caused by Cactus virus X (CVX), spread through infected cutting tools, sap contact, and infected planting material. Purely viral — no fungal or bacterial element.",
        "organic": "No cure exists once infected. Remove and destroy infected plants. Sterilize all cutting tools with 70% alcohol or 10% bleach between uses. Source virus-free cuttings only.",
        "chemical": "There is no chemical treatment for viral infections. Prevent spread by disinfecting tools regularly, avoiding replanting in infected soil, and using certified disease-free planting stock."
    },
    "Healthy Stem": {
        "cause": "No disease detected. The plant appears healthy with no signs of infection, fungal growth, or lesions.",
        "organic": "Maintain health with regular neem oil sprays (preventive). Ensure proper drainage and avoid overwatering. Mulch around the base to retain moisture and reduce soil-borne pathogens.",
        "chemical": "No treatment needed. A light copper-based fungicide spray once a month during the rainy season can help as a preventive measure against fungal infections."
    },
    "Brown Spot": {
        "cause": "Caused by the fungus Botryosphaeria dothidea or Alternaria species. Triggered by high humidity, poor drainage, and waterlogged soil. Appears as round brown spots with yellow halos.",
        "organic": "Spray neem oil solution (10 ml per litre) every 7 days. Improve soil drainage and reduce irrigation frequency. Remove infected leaves and stems. Apply compost to boost plant immunity.",
        "chemical": "Apply Iprodione (0.2%) or Carbendazim (0.1%) fungicide every 10 days. Remove and burn heavily infected parts before applying treatment for best results."
    },
    "Soft Rot": {
        "cause": "Caused by the bacterium Pectobacterium carotovorum or fungus Pythium species. Spreads rapidly in waterlogged conditions and through wounds. Causes water-soaked, mushy, foul-smelling lesions.",
        "organic": "Reduce irrigation immediately. Apply Trichoderma harzianum bio-fungicide to soil. Remove rotted tissue and dust cut surfaces with ash or turmeric powder to dry them out.",
        "chemical": "Apply Copper hydroxide (0.3%) bactericide/fungicide to affected areas. In severe cases, drench soil with Metalaxyl (0.2%). Ensure good drainage to prevent recurrence."
    },
    "Black Spot": {
        "cause": "Caused by the fungus Pseudocercospora species or Alternaria alternata. Promoted by high humidity and warm temperatures. Appears as small black raised spots that expand and merge over time.",
        "organic": "Spray diluted apple cider vinegar (1:10 ratio with water) or baking soda solution weekly. Apply neem oil as a preventive. Ensure good air circulation through regular pruning.",
        "chemical": "Use Tebuconazole (0.1%) or Propiconazole (0.1%) fungicide every 14 days. Mancozeb (0.25%) is an affordable alternative during high-humidity periods."
    },
    "Gray Blight": {
        "cause": "Caused by the fungus Pestalotiopsis species. Common in poorly ventilated farms with high humidity. Begins as grey water-soaked spots that turn ash-grey with dark margins as they expand.",
        "organic": "Prune affected parts and improve air circulation. Apply Bordeaux mixture (copper sulfate + lime) every 10–14 days as a traditional organic remedy. Avoid wetting foliage during irrigation.",
        "chemical": "Apply Chlorothalonil (0.2%) or Mancozeb + Metalaxyl combination fungicide every 10 days. Alternate between fungicides to prevent resistance buildup."
    }
}

# ================= IMAGE PROCESSING ================= #

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.medianBlur(img, 3)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = preprocess_input(img)
    return img


def predict_stem(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)

    features    = feature_extractor1.predict(img)
    img_lbp     = cv2.resize(cv2.imread(image_path), (224, 224))
    lbp_features = extract_lbp(img_lbp).reshape(1, -1)
    features    = np.hstack((features, lbp_features))

    prediction  = svm_model1.predict(features)
    disease     = label_encoder1.inverse_transform(prediction)
    return disease[0]


def predict_spot(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)

    features   = feature_extractor2.predict(img)
    prediction = svm_model2.predict(features)
    disease    = label_encoder2.inverse_transform(prediction)
    return disease[0]


# ================= AUTH ================= #

@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect("/")
        else:
            message = "Invalid credentials"
    return render_template("login.html", message=message)


@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        name     = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            message = "User already exists"
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password, name=name)
            db.session.add(new_user)
            db.session.commit()
            return redirect("/login")
    return render_template("register.html", message=message)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")


# ================= MAIN ================= #

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file       = request.files["image"]
        model_type = request.form["model"]

        filename = file.filename
        filepath = os.path.join("static/uploads", filename)
        file.save(filepath)

        # Run prediction
        if model_type == "stem":
            result = predict_stem(filepath)
        else:
            result = predict_spot(filepath)

        # Save to database
        new_prediction = Prediction(
            user_id=current_user.id,
            image_name=filename,
            result=result
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Return JSON (used by the AJAX form in index.html)
        return jsonify({
            "result": result,
            "image": filename
        })

    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", predictions=user_predictions)


@app.route("/result")
@login_required
def result():
    image  = request.args.get("image")
    result = request.args.get("result")
    info   = DISEASE_INFO.get(result.title() if result else "", {})
    return render_template("result.html", image=image, result=result, info=info)


# ================= RUN ================= #

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)