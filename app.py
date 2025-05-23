from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from collections import Counter
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Transliterator Class ---
class Transliterator:
    def __init__(self):
        self.map = {
            'અ': 'a', 'આ': 'aa', 'ઇ': 'i', 'ઈ': 'ii', 'ઉ': 'u', 'ઊ': 'uu', 'ઋ': 'ri',
            'એ': 'e', 'ઐ': 'ai', 'ઓ': 'o', 'ઔ': 'au',
            'ક': 'k', 'ખ': 'kh', 'ગ': 'g', 'ઘ': 'gh', 'ચ': 'ch', 'છ': 'chh', 'જ': 'j', 'ઝ': 'jh',
            'ટ': 't', 'ઠ': 'th', 'ડ': 'd', 'ઢ': 'dh', 'ણ': 'n', 'ત': 't', 'થ': 'th', 'દ': 'd',
            'ધ': 'dh', 'ન': 'n', 'પ': 'p', 'ફ': 'ph', 'બ': 'b', 'ભ': 'bh', 'મ': 'm',
            'ય': 'y', 'ર': 'r', 'લ': 'l', 'વ': 'v', 'શ': 'sh', 'ષ': 'sh', 'સ': 's', 'હ': 'h',
            'ળ': 'l', 'ક્ષ': 'ksh', 'જ્ઞ': 'gn'
        }

    def gujarati_to_english(self, text):
        output = ""
        for char in text:
            output += self.map.get(char, char)
        return output

# --- Model Class Definitions ---
class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _gini(self, y):
        counts = Counter(y)
        return 1 - sum((c / len(y)) ** 2 for c in counts.values())

    def _best_split(self, X, y):
        best_gain = -1
        best_feat, best_val = None, None
        current_gini = self._gini(y)
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for val in values:
                left_idx = X[:, feature] <= val
                right_idx = X[:, feature] > val
                if sum(left_idx) < self.min_samples_split or sum(right_idx) < self.min_samples_split:
                    continue
                left = y[left_idx]
                right = y[right_idx]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = current_gini - (
                    len(left)/len(y)*self._gini(left) + len(right)/len(y)*self._gini(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_val = feature, val
        return best_feat, best_val

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]
        feature, value = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        left_idx = X[:, feature] <= value
        right_idx = X[:, feature] > value
        left_branch = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_branch = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return (feature, value, left_branch, right_branch)

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, value, left, right = node
        if x[feature] <= value:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

class KNNScratch:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        preds = []
        for x in X:
            distances = [self._euclidean(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(preds)

# Instantiate Transliterator
transliterator = Transliterator()

# Load the trained models
models = {}
model_names = ["decision_tree", "knn"]
for name in model_names:
    try:
        with open(f"{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)
        logger.info(f"Successfully loaded {name}_model.pkl")
    except Exception as e:
        logger.error(f"Failed to load {name}_model.pkl: {str(e)}")
        raise

# Load the dataset to get category mappings
def load_category_mappings(filename="career_data_new.csv"):
    try:
        df = pd.read_csv(filename)
        skill_mapping = dict(enumerate(df['Skill'].astype('category').cat.categories))
        interest_mapping = dict(enumerate(df['Interests'].astype('category').cat.categories))
        experience_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
        logger.info("Successfully loaded category mappings from dataset")
        logger.debug(f"Skill mapping: {skill_mapping}")
        logger.debug(f"Interest mapping: {interest_mapping}")
        return skill_mapping, interest_mapping, experience_mapping
    except FileNotFoundError:
        logger.warning("career_data_new.csv not found, using fallback mappings")
        # Fallback mappings (replace with actual categories from your dataset)
        fallback_mappings = (
            {0: 'Python', 1: 'Java', 2: 'AI', 3: 'C++', 4: 'JavaScript'},  # TODO: Replace
            {0: 'AI Research', 1: 'Web Development', 2: 'Data Science', 3: 'Cybersecurity', 4: 'Mobile Development'},  # TODO: Replace
            {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
        )
        logger.debug(f"Fallback skill mapping: {fallback_mappings[0]}")
        logger.debug(f"Fallback interest mapping: {fallback_mappings[1]}")
        return fallback_mappings
    except Exception as e:
        logger.error(f"Error loading category mappings: {str(e)}")
        raise

skill_mapping, interest_mapping, experience_mapping = load_category_mappings()

# Helper function to preprocess input
def preprocess_input(skills, interests, experience):
    try:
        # Validate inputs
        skills = skills.strip().lower() if skills and isinstance(skills, str) else "python"
        interests = interests.strip().lower() if interests and isinstance(interests, str) else "ai research"
        experience = experience if experience in experience_mapping else "Beginner"

        # Map to numerical codes using exact matching
        skill_code = None
        for k, v in skill_mapping.items():
            if v.lower() == skills:
                skill_code = k
                break
        if skill_code is None:
            logger.warning(f"Skill '{skills}' not found in skill_mapping, defaulting to code 0")
            skill_code = 0  # Default to first category

        interest_code = None
        for k, v in interest_mapping.items():
            if v.lower() == interests:
                interest_code = k
                break
        if interest_code is None:
            logger.warning(f"Interest '{interests}' not found in interest_mapping, defaulting to code 0")
            interest_code = 0  # Default to first category

        experience_code = experience_mapping[experience]

        # Log the mappings used
        logger.debug(f"Input: skills='{skills}', mapped to '{skill_mapping.get(skill_code, 'Unknown')}' (code: {skill_code})")
        logger.debug(f"Input: interests='{interests}', mapped to '{interest_mapping.get(interest_code, 'Unknown')}' (code: {interest_code})")
        logger.debug(f"Input: experience='{experience}', mapped to code: {experience_code}")

        return np.array([[skill_code, interest_code, experience_code]], dtype=float)
    except Exception as e:
        logger.error(f"Error in preprocess_input: {str(e)}")
        raise

# Route to serve the frontend
@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template('index.html')

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        logger.debug(f"Received request: {data}")

        # Extract and validate inputs
        skills = data.get('skills', '')
        interests = data.get('interests', '')
        experience = data.get('experience', 'Beginner')
        language = data.get('language', 'english')

        # Preprocess the input
        input_data = preprocess_input(skills, interests, experience)

        # Get predictions from KNN and Decision Tree models
        predictions = {}
        for name in ['decision_tree', 'knn']:
            try:
                model = models[name]
                raw_pred = model.predict(input_data)[0]
                logger.debug(f"Raw {name} prediction: {raw_pred}")
                # Transliterate only if language is Gujarati and prediction contains Gujarati script
                pred = raw_pred
                if language == 'gujarati' and any(char in transliterator.map for char in str(raw_pred)):
                    pred = transliterator.gujarati_to_english(raw_pred)
                    logger.debug(f"Transliterated {name} prediction: {pred}")
                predictions[name] = str(pred)
            except Exception as e:
                logger.error(f"Error in {name} prediction: {str(e)}")
                raise

        # Determine consensus prediction
        consensus = Counter(predictions.values()).most_common(1)[0][0]
        raw_consensus = consensus
        if language == 'gujarati' and any(char in transliterator.map for char in consensus):
            consensus = transliterator.gujarati_to_english(consensus)
            logger.debug(f"Transliterated consensus: {consensus}")

        logger.info(f"Consensus prediction: {consensus} (raw: {raw_consensus})")

        # Format predictions for response
        prediction_list = [f"{name.replace('_', ' ').title()}: {pred}" for name, pred in predictions.items()]

        return jsonify({
            'consensus': consensus,
            'predictions': prediction_list
        })

    except Exception as e:
        logger.error(f"Error in /recommend: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'An error occurred while processing the recommendation: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)