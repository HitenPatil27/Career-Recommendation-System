# Career Recommendation System

**Choose Your Ideal Career**  
This Flask-based bilingual web application uses machine learning to recommend the best career path based on your skills, interests, and experience level.

---

## Overview

The Career Recommendation System predicts careers such as **Data Scientist** or **Web Developer** based on your input using custom-built machine learning models from scratch — **K-Nearest Neighbors (KNN)** and **Decision Tree** — trained on real-world data.

---

## Features

- **Dual Model Prediction**: Combines predictions from KNN and Decision Tree.
- **Bilingual Support**: English and Gujarati interface.
- **Custom ML from Scratch**: No external libraries like scikit-learn used.
- **Input Validation**: Logs and handles mismatched inputs with default fallback codes.
- **Model Accuracy**:
  - KNN: **99%**
  - Decision Tree: **92%**

---

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **ML Models**: KNN, Decision Tree (hand-coded)
- **Dataset**: `career_data_new.csv`

---

## File Structure
  project/
  │
  ├── app.py # Flask app backend
  ├── career_data_new.csv # Dataset
  ├── decision_tree_model.pkl # Trained Decision Tree model
  ├── knn_model.pkl # Trained KNN model
  │
  ├── templates/
  │ └── index.html # Frontend HTML
  │
  ├── static/
  │ ├── styles.css # Frontend styling
  │ └── script.js # Frontend behavior
  │
  └── app.log # Logs (debug, info, errors)


---

## How to Run

1. **Clone this repo**:
   ```bash
   git clone https://github.com/your-username/career-recommendation-system.git
   cd career-recommendation-system
2. **Install dependencies**:
  pip install flask pandas numpy

3. **Run the app**:
   python app.py
---
