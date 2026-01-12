Got it.
Below is ONE SINGLE FILE, nothing else.
Copy–paste this entire thing as README.md.

⸻


# FinZen — Helping Gen-Z Track Expenses

FinZen is a Streamlit-based personal finance tracking application designed for Gen-Z users. It combines expense tracking, basic machine-learning–based categorization, gamification via a virtual pet, and a lightweight chatbot, all backed by MongoDB for persistence.

---

## Features

### Expense Tracking
- Add and store daily expenses
- Automatic classification into **Needs** and **Wants**
- Category prediction (Food, Utilities, Transport, Shopping, etc.)

### Machine Learning
- Text-based expense classification using `CountVectorizer`
- RandomForest and Naive Bayes classifiers
- Automatically trains models if pickle files are missing
- Keyword-based fallback categorization for reliability

### FinPet (Gamification)
- Virtual pet that grows as the user saves
- XP / level system based on savings (`zen_savings`)
- GIF-based visual feedback

### Chatbot
- Rule-based financial assistant
- Answers queries related to:
  - Expense history
  - Weekly wants
  - Remaining funds
  - FinPet status
  - Basic finance questions

### Database
- MongoDB backend
- Collections for users, transactions, and user data
- Passwords hashed using SHA-256

---

## Project Structure

finzen/
├── app1.py                     # Main Streamlit application
├── requirements.txt            # Python dependencies
├── gifs/                       # FinPet GIF assets
├── vectorizer.pkl              # (optional) text vectorizer
├── type_classifier.pkl         # (optional) needs/wants classifier
├── cat_classifier.pkl          # (optional) category classifier
├── vectorizer_needs.pkl        # (optional) needs-category vectorizer
├── needs_cat_classifier.pkl    # (optional) needs category classifier


## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Database:** MongoDB  
- **Machine Learning:** scikit-learn  
- **Data Handling:** pandas, numpy  
- **Utilities:** Pillow, certifi  

---

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/sovopr/finzen.git
cd finzen

2. (Recommended) Create a virtual environment

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3. Install dependencies

pip install -r requirements.txt

For a minimal setup:

pip install streamlit scikit-learn pandas pymongo numpy pillow certifi


⸻

MongoDB Configuration

The MongoDB connection string is defined inside app1.py.

You must provide:
	•	MongoDB username
	•	MongoDB password
	•	MongoDB cluster URL

Example (do not commit credentials):

username_db = quote_plus("YOUR_USERNAME")
password_db = quote_plus("YOUR_PASSWORD")


⸻

Running the Application

streamlit run app1.py

On first run:
	•	ML models are trained if pickle files are missing
	•	Trained models are saved locally for future use

⸻

Known Limitations
	•	Password hashing uses SHA-256 without salting
	•	ML models are trained on a small synthetic dataset
	•	Entire application logic resides in a single file (app1.py)
	•	Secrets are not managed via environment variables
	•	No automated tests or CI pipeline

⸻

Future Improvements
	•	Modularize the codebase
	•	Move model training to a separate script
	•	Use .env files for configuration
	•	Upgrade password hashing (bcrypt / argon2)
	•	Add Docker support
	•	Improve ML accuracy with real user data
	•	Add unit tests and CI/CD
