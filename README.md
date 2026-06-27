#  Dragon Fruit Disease Detection System

An AI-powered web application that detects diseases in dragon fruit stems using Machine Learning and Deep Learning techniques. The system allows users to upload an image of a dragon fruit stem and predicts the disease with high accuracy, helping farmers identify infections early.

---

## 📌 Features

- User Registration and Login
- Upload Dragon Fruit Stem Images
- AI-Based Disease Detection
- Displays Predicted Disease
- Prediction History
- Secure User Authentication
- Simple and Responsive Interface

---

## 🦠 Diseases Detected

The system can detect the following conditions:

- Healthy Stem
- Brown Spot
- Stem Canker
- Anthracnose
- Grey Blight
- Black Spot
- Soft Rot
- Cactus Virus X

---

## 🛠 Technologies Used

### Frontend
- HTML
- CSS

### Backend
- Python
- Flask

### Machine Learning / Deep Learning
- TensorFlow
- Keras
- Scikit-learn

### Database
- SQLite

---

## 📂 Project Structure

```
Dragon-Fruit-Disease-Detection-System/
│
├── static/
├── templates/
├── app.py
├── models.py
├── feature_extractor_model1.h5
├── feature_extractor_model2.h5
├── svm_model1.pkl
├── svm_model2.pkl
├── label_encoder_model1.pkl
├── label_encoder_model2.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/MonaV7/Dragon-Fruit-Disease-Detection-System.git
```

Move into the project folder:

```bash
cd Dragon-Fruit-Disease-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 📸 How It Works

1. Register or Login.
2. Upload a dragon fruit stem image.
3. The system extracts image features.
4. The trained model predicts the disease.
5. The result is displayed along with prediction history.

---

## 🎯 Future Enhancements

- Mobile Application
- Cloud Deployment
- Real-time Camera Detection
- Disease Severity Analysis
- Treatment Recommendation System

---

## 👩‍💻 Author

**Monanvitha**

GitHub: https://github.com/MonaV7

---

## 📄 License

This project is licensed under the MIT License.
