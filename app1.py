import os
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
from hashlib import sha256
from datetime import datetime, timedelta
import pickle
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from urllib.parse import quote_plus
import certifi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import numpy as np
import pytz
from datetime import datetime, timedelta
import pytz

# Helper function: convert a datetime (or ISO string) to a naive datetime
def to_naive(dt):
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt

# Calculate start_of_week (naive datetime)
start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())

# Localize start_of_week to UTC (only for initial calculation, not used in comparisons)
start_of_week_utc = pd.to_datetime(start_of_week).tz_localize('UTC')

# Mock Data for Paytm Transactions
paytm_mock_data = [
    {"description": "Bought coffee", "total_amount": 150, "category": "Food", "date": datetime.now() - timedelta(days=1)},
    {"description": "Shopping on Amazon", "total_amount": 2000, "category": "Shopping", "date": datetime.now() - timedelta(days=2)},
    {"description": "Uber ride to office", "total_amount": 300, "category": "Transport", "date": datetime.now() - timedelta(days=3)},
    {"description": "Booked movie tickets", "total_amount": 500, "category": "Entertainment", "date": datetime.now() - timedelta(days=4)},
    {"description": "Grocery shopping at BigBasket", "total_amount": 1200, "category": "Shopping", "date": datetime.now() - timedelta(days=5)},
]

# Train a simple AI model to predict categories based on descriptions
categories = ["Food", "Shopping", "Transport", "Entertainment"]
descriptions = [transaction["description"] for transaction in paytm_mock_data]
labels = [transaction["category"] for transaction in paytm_mock_data]

vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(descriptions, labels)

# ---------- MODEL LOADING / TRAINING ----------
if (os.path.exists("vectorizer.pkl") and 
    os.path.exists("type_classifier.pkl") and 
    os.path.exists("cat_classifier.pkl") and 
    os.path.exists("vectorizer_needs.pkl") and 
    os.path.exists("needs_cat_classifier.pkl")):
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("type_classifier.pkl", "rb") as f:
        type_clf = pickle.load(f)
    with open("cat_classifier.pkl", "rb") as f:
        cat_clf = pickle.load(f)
    with open("vectorizer_needs.pkl", "rb") as f:
        vectorizer_needs = pickle.load(f)
    with open("needs_cat_classifier.pkl", "rb") as f:
        needs_cat_clf = pickle.load(f)
else:
    def train_models():
        data = {
            "description": [
                "Bought milk and bread", "Ordered pizza online", "Had dinner at an Italian restaurant", 
                "Grabbed a coffee on the way", "Lunch at a local cafe", "Grocery shopping for vegetables and fruits", 
                "Dinner at a sushi bar", "Breakfast at a diner", "Snacked on chips", "Ordered takeout Chinese food", 
                "Paid electricity bill for this month", "Paid water bill", "Settled internet bill", 
                "Paid gas bill", "Paid cable TV subscription", "Received phone bill", "Paid heating bill", 
                "Paid property tax", "Monthly rent payment", "Paid maintenance fee for condo", 
                "Bought a monthly bus pass", "Uber ride to the airport", "Taxi fare from downtown", 
                "Subway ticket purchase", "Train ticket to the city", "Rented a car for a day", 
                "Bike sharing rental", "Paid for ride-sharing service", "Bus fare for school commute", 
                "Ferry ticket to the island", "Bought new shoes online", "Purchased a designer bag", 
                "Online shopping for clothes", "Bought a new jacket at the mall", "Purchased a smartphone accessory", 
                "Bought electronics from a store", "Shopping spree at a department store", "Bought a new pair of jeans", 
                "Purchased a watch", "Bought home decor items", "Subscribed to an online course", 
                "Bought textbooks for college", "Paid tuition fees", "Enrolled in a language course", 
                "Paid for a workshop", "Purchased educational software", "Registered for an online seminar", 
                "Bought study materials", "Paid for certification exam", "Subscribed to an academic journal", 
                "Movie night ticket", "Concert ticket purchase", "Attended a comedy show", 
                "Paid for streaming service subscription", "Bought a ticket for a theatre play", 
                "Went to a music festival", "Paid for a dance class", "Attended a sports game", 
                "Bought a video game", "Visited an amusement park", "Recharged my mobile phone", 
                "Bought a birthday gift for a friend", "Repaired a broken laptop", "Gym membership fee", 
                "Paid for a haircut", "Purchased office supplies", "Paid for a pet grooming session", 
                "Donated to charity", "Purchased a book", "Had a medical checkup"
            ],
            "type": ["Needs", "Wants", "Wants", "Wants", "Needs", "Needs", "Wants", "Needs", "Wants", "Wants",
                     "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs",
                     "Needs", "Wants", "Wants", "Needs", "Needs", "Wants", "Needs", "Wants", "Needs", "Wants",
                     "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants",
                     "Wants", "Needs", "Needs", "Wants", "Needs", "Needs", "Wants", "Needs", "Needs", "Wants",
                     "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants",
                     "Needs", "Wants", "Needs", "Wants", "Needs", "Needs", "Wants", "Wants", "Needs", "Needs"],
            "category": ["Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food",
                         "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", 
                         "Utilities", "Housing", "Housing", "Transport", "Transport", "Transport", "Transport", 
                         "Transport", "Transport", "Transport", "Transport", "Transport", "Transport", "Shopping", 
                         "Shopping", "Shopping", "Shopping", "Shopping", "Electronics", "Shopping", "Shopping", 
                         "Shopping", "Shopping", "Education", "Education", "Education", "Education", "Education", 
                         "Education", "Education", "Education", "Education", "Education", "Entertainment", 
                         "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment", 
                         "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment", 
                         "Utilities", "Gifts", "Electronics", "Fitness", "Personal Care", "Shopping", "Personal Care", 
                         "Charity", "Education", "Health"]
        }
        df = pd.DataFrame(data)
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df["description"])
        
        y_type = df["type"]
        X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
        type_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        type_clf.fit(X_train_type, y_train_type)
        
        y_cat = df["category"]
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        cat_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cat_clf.fit(X_train_cat, y_train_cat)
        
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open("type_classifier.pkl", "wb") as f:
            pickle.dump(type_clf, f)
        with open("cat_classifier.pkl", "wb") as f:
            pickle.dump(cat_clf, f)
    
    def train_needs_model():
        data = {
            "description": [
                "Bought milk and bread", "Ordered pizza online", "Had dinner at an Italian restaurant", 
                "Grabbed a coffee on the way", "Lunch at a local cafe", "Grocery shopping for vegetables and fruits", 
                "Dinner at a sushi bar", "Breakfast at a diner", "Snacked on chips", "Ordered takeout Chinese food", 
                "Paid electricity bill for this month", "Paid water bill", "Settled internet bill", 
                "Paid gas bill", "Paid cable TV subscription", "Received phone bill", "Paid heating bill", 
                "Paid property tax", "Monthly rent payment", "Paid maintenance fee for condo", 
                "Bought a monthly bus pass", "Uber ride to the airport", "Taxi fare from downtown", 
                "Subway ticket purchase", "Train ticket to the city", "Rented a car for a day", 
                "Bike sharing rental", "Paid for ride-sharing service", "Bus fare for school commute", 
                "Ferry ticket to the island", "Bought new shoes online", "Purchased a designer bag", 
                "Online shopping for clothes", "Bought a new jacket at the mall", "Purchased a smartphone accessory", 
                "Bought electronics from a store", "Shopping spree at a department store", "Bought a new pair of jeans", 
                "Purchased a watch", "Bought home decor items", "Subscribed to an online course", 
                "Bought textbooks for college", "Paid tuition fees", "Enrolled in a language course", 
                "Paid for a workshop", "Purchased educational software", "Registered for an online seminar", 
                "Bought study materials", "Paid for certification exam", "Subscribed to an academic journal", 
                "Movie night ticket", "Concert ticket purchase", "Attended a comedy show", 
                "Paid for streaming service subscription", "Bought a ticket for a theatre play", 
                "Went to a music festival", "Paid for a dance class", "Attended a sports game", 
                "Bought a video game", "Visited an amusement park", "Recharged my mobile phone", 
                "Bought a birthday gift for a friend", "Repaired a broken laptop", "Gym membership fee", 
                "Paid for a haircut", "Purchased office supplies", "Paid for a pet grooming session", 
                "Donated to charity", "Purchased a book", "Had a medical checkup"
            ],
            "type": ["Needs", "Wants", "Wants", "Wants", "Needs", "Needs", "Wants", "Needs", "Wants", "Wants",
                     "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs", "Needs",
                     "Needs", "Wants", "Wants", "Needs", "Needs", "Wants", "Needs", "Wants", "Needs", "Wants",
                     "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants",
                     "Wants", "Needs", "Needs", "Wants", "Needs", "Needs", "Wants", "Needs", "Needs", "Wants",
                     "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants", "Wants",
                     "Needs", "Wants", "Needs", "Wants", "Needs", "Needs", "Wants", "Wants", "Needs", "Needs"],
            "category": ["Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food", "Food",
                         "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", "Utilities", 
                         "Utilities", "Housing", "Housing", "Transport", "Transport", "Transport", "Transport", 
                         "Transport", "Transport", "Transport", "Transport", "Transport", "Transport", "Shopping", 
                         "Shopping", "Shopping", "Shopping", "Shopping", "Electronics", "Shopping", "Shopping", 
                         "Shopping", "Shopping", "Education", "Education", "Education", "Education", "Education", 
                         "Education", "Education", "Education", "Education", "Education", "Entertainment", 
                         "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment", 
                         "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment", 
                         "Utilities", "Gifts", "Electronics", "Fitness", "Personal Care", "Shopping", "Personal Care", 
                         "Charity", "Education", "Health"]
        }
        df_full = pd.DataFrame(data)
        df_needs = df_full[df_full["type"] == "Needs"].reset_index(drop=True)
        
        vectorizer_needs = CountVectorizer(stop_words="english")
        X = vectorizer_needs.fit_transform(df_needs["description"])
        y_needs = df_needs["category"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_needs, test_size=0.2, random_state=42)
        needs_cat_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        needs_cat_clf.fit(X_train, y_train)
        
        with open("vectorizer_needs.pkl", "wb") as f:
            pickle.dump(vectorizer_needs, f)
        with open("needs_cat_classifier.pkl", "wb") as f:
            pickle.dump(needs_cat_clf, f)
    
    train_models()
    train_needs_model()
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("type_classifier.pkl", "rb") as f:
        type_clf = pickle.load(f)
    with open("cat_classifier.pkl", "rb") as f:
        cat_clf = pickle.load(f)
    with open("vectorizer_needs.pkl", "rb") as f:
        vectorizer_needs = pickle.load(f)
    with open("needs_cat_classifier.pkl", "rb") as f:
        needs_cat_clf = pickle.load(f)

# ---------- MongoDB Remote Connection Setup ----------
username_db = quote_plus("soveetprusty")
password_db = quote_plus("@Noobdamaster69")
conn_str = (
    f"mongodb+srv://{username_db}:{password_db}"
    "@cluster0.bjzstq0.mongodb.net/?retryWrites=true&w=majority"
)

client = pymongo.MongoClient(
    conn_str,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=30000
)
db = client["agri_app"]
users_col = db["users"]
transactions_col = db["transactions"]
data_col = db["user_data"]

# ---------- Session State Initialization ----------
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "page" not in st.session_state:
    st.session_state.page = "Login"

# ---------- Helper Functions ----------
def safe_rerun():
    st.rerun()

def hash_password(password: str) -> str:
    return sha256(password.encode('utf-8')).hexdigest()

def authenticate_user(username: str, password: str) -> bool:
    user = users_col.find_one({"username": username})
    return user and user["password"] == hash_password(password)

def register_user(username: str, password: str) -> bool:
    if users_col.find_one({"username": username}):
        return False
    users_col.insert_one({"username": username, "password": hash_password(password)})
    data_col.insert_one({
        "username": username,
        "expenses": [],
        "goals": [],
        "available_funds": 0,
        "budget_limits": {"Wants": 0, "Needs": 0},
        "pet_level": 1,
        "rewards": [],
        "zen_savings": 0,
        "zen_mode": False,
        "last_weekly_reset": datetime.now(),
        "tracking_enabled": True
    })
    return True

def check_pet_level_up():
    user_data = get_user_data()
    if not user_data:
        return
    
    zen_savings = user_data.get("zen_savings", 0)
    current_level = user_data.get("pet_level", 1)
    rewards = user_data.get("rewards", [])
    updated = False

    while True:
        required_xp = current_level * 500
        if zen_savings >= required_xp:
            levels_gained = zen_savings // required_xp
            if levels_gained == 0:
                break
                
            new_level = current_level + levels_gained
            zen_savings %= required_xp
            updated = True
            
            for lvl in range(current_level + 1, new_level + 1):
                cashback = lvl * 50
                rewards.append(f"Level {lvl} Reward: ₹{cashback} cashback!")
                if lvl % 3 == 0:
                    rewards.append(f"🎉 Bonus Reward at Level {lvl}!")
            
            current_level = new_level
            st.toast(f"🎉 Leveled up {levels_gained} times! Now Level {new_level}")
        else:
            break

    if updated:
        update_user_data({
            "pet_level": current_level,
            "rewards": rewards,
            "zen_savings": zen_savings
        })
        safe_rerun()

def get_pet_image(lvl):
    # Check if image file exists; if not, return None with status text.
    if lvl < 5:
        return None, "🥚 Egg (Hatch at Level 5)"
    elif lvl < 15:
        pet_image = "gifs/gif1.gif"
        return pet_image if os.path.exists(pet_image) else None, "Ghastly"
    elif lvl < 25:
        pet_image = "gifs/gif2.gif"
        return pet_image if os.path.exists(pet_image) else None, "Haunter"
    else:
        pet_image = "gifs/gif3.gif"
        return pet_image if os.path.exists(pet_image) else None, "Gengar"

def show_finpet():
    check_pet_level_up()
    
    st.title("🐾 Your FinPet")
    user_data = get_user_data()
    if not user_data:
        st.error("No user data found")
        return
    
    level = user_data.get("pet_level", 1)
    zen_savings = user_data.get("zen_savings", 0)
    rewards = user_data.get("rewards", [])
    
    required_xp = level * 500
    current_progress = zen_savings / required_xp if required_xp > 0 else 0

    pet_image, pet_status = get_pet_image(level)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if pet_image:
            st.image(pet_image, width=200)
        else:
            st.markdown(f"<h2 style='text-align: center;'>{pet_status}</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### Level {level}")
        st.progress(min(current_progress, 1.0))
        st.write(f"*XP:* {zen_savings}/{required_xp}")
        st.write(f"*Next Level:* ₹{max(0, required_xp - zen_savings)} needed")

    st.subheader("🎁 Recent Rewards")
    for reward in rewards[-3:]:
        st.markdown(f"- {reward}")

def get_user_data():
    user_data = data_col.find_one({"username": st.session_state.current_user})
    if user_data:
        # Ensure last_weekly_reset is a naive datetime
        if isinstance(user_data.get("last_weekly_reset"), str):
            user_data["last_weekly_reset"] = to_naive(user_data["last_weekly_reset"])
        elif isinstance(user_data.get("last_weekly_reset"), datetime):
            user_data["last_weekly_reset"] = to_naive(user_data["last_weekly_reset"])
        
        for expense in user_data.get("expenses", []):
            if isinstance(expense.get("date"), str):
                expense["date"] = to_naive(expense["date"])
            elif isinstance(expense.get("date"), datetime):
                expense["date"] = to_naive(expense["date"])
        
        last_reset = user_data.get("last_weekly_reset", datetime.now())
        if datetime.now() - last_reset > timedelta(weeks=1):
            weekly_limit = user_data.get("budget_limits", {}).get("Wants", 0)
            start_of_week_local = last_reset - timedelta(days=last_reset.weekday())
            expenses = user_data.get("expenses", [])
            weekly_spent = sum(e["total_amount"] for e in expenses 
                               if e.get("type") == "Wants" and to_naive(e.get("date")) >= start_of_week_local)
            unused_wants = max(0, weekly_limit - weekly_spent)
            new_savings = user_data.get("zen_savings", 0) + unused_wants
            rewards = user_data.get("rewards", [])
            if unused_wants > 0:
                cashback = round(unused_wants * 0.1, 2)
                rewards.append(f"Weekly Cashback: ₹{cashback} awarded for unused 'Wants' budget!")
            update_user_data({
                "zen_savings": new_savings,
                "last_weekly_reset": datetime.now(),
                "rewards": rewards
            })
            check_pet_level_up()
        
        if "tracking_enabled" not in user_data:
            data_col.update_one(
                {"username": st.session_state.current_user},
                {"$set": {"tracking_enabled": True}}
            )
            user_data["tracking_enabled"] = True
            
        return user_data
    return None

def update_user_data(update: dict):
    def convert_np_ints(value):
        if isinstance(value, dict):
            return {key: convert_np_ints(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [convert_np_ints(item) for item in value]
        elif isinstance(value, (int, float, bool, str)):
            return value
        elif isinstance(value, np.int64):
            return int(value)
        return value

    update = convert_np_ints(update)
    data_col.update_one({"username": st.session_state.current_user}, {"$set": update})

def fallback_category(description, predicted_category):
    desc = description.lower()
    food_keywords = ["dinner", "lunch", "breakfast", "meal", "snack", "pizza", "burger", "sushi", "coffee", "tea"]
    if any(word in desc for word in food_keywords):
        return "Food"
    utilities_keywords = ["electricity", "water", "gas", "internet", "cable", "phone bill", "rent", "tax", "maintenance"]
    if any(word in desc for word in utilities_keywords):
        return "Utilities"
    transport_keywords = ["bus", "taxi", "uber", "subway", "train", "ferry", "ride", "car", "bike"]
    if any(word in desc for word in transport_keywords):
        return "Transport"
    shopping_keywords = ["shoes", "bag", "clothes", "jacket", "accessory", "electronics", "watch", "jeans", "decor"]
    if any(word in desc for word in shopping_keywords):
        return "Shopping"
    education_keywords = ["course", "textbook", "tuition", "language", "workshop", "software", "seminar", "study", "certification", "journal"]
    if any(word in desc for word in education_keywords):
        return "Education"
    entertainment_keywords = ["movie", "concert", "comedy", "streaming", "theatre", "festival", "dance", "sports", "game", "amusement"]
    if any(word in desc for word in entertainment_keywords):
        return "Entertainment"
    misc_keywords = ["mobile", "gift", "repair", "gym", "haircut", "office", "grooming", "donate", "book", "medical", "medicine"]
    if any(word in desc for word in misc_keywords):
        return "Miscellaneous"
    return "Miscellaneous"

def show_chatbot():
    st.title("💬 AI Chatbot")
    st.markdown("Ask me anything about your finances, FinPet, expense history, available funds, goals, weekly wants, or even about your FinPet level!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.markdown(f"You: {chat['message']}")
        else:
            st.markdown(f"Chatbot: {chat['message']}")
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:")
        submit = st.form_submit_button("Send")
        if submit and user_input:
            st.session_state.chat_history.append({"sender": "user", "message": user_input})
            response = generate_ai_response(user_input)
            st.session_state.chat_history.append({"sender": "chatbot", "message": response})
            safe_rerun()

def generate_ai_response(message):
    msg = message.lower()
    user_data = get_user_data()

    greeting_triggers = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(trigger in msg for trigger in greeting_triggers):
        return "Hello! How can I help you with your finances today?"
    if "how are you" in msg:
        return "I'm just a chatbot, but I'm here to help you manage your money and FinPet!"

    finpet_triggers = ["finpet", "pet level", "what level", "level of my pet", "fin pet"]
    if any(trigger in msg for trigger in finpet_triggers):
        if user_data:
            level = user_data.get("pet_level", 1)
            return f"Your FinPet is currently at level {level}."
        else:
            return "I couldn't fetch your FinPet info. Please try again."

    expense_history_triggers = ["expense history", "my expenses", "what did i spend", "recorded expenses", "expense records"]
    if any(trigger in msg for trigger in expense_history_triggers):
        if user_data:
            expenses = user_data.get("expenses", [])
            if not expenses:
                return "You have not recorded any expenses yet."
            else:
                response_str = "Here is your complete expense history:\n"
                for expense in expenses:
                    dt = to_naive(expense.get("date"))
                    desc = expense.get("description", "No description")
                    amt = expense.get("total_amount", 0)
                    typ = expense.get("type", "Unknown")
                    response_str += f"- {dt}: '{desc}' costing ₹{amt} ({typ})\n"
                return response_str
        else:
            return "I couldn't fetch your expense history. Please try again."

    last_expense_triggers = ["last expense", "most recent expense", "what is my expense", "tell me my expense"]
    if any(trigger in msg for trigger in last_expense_triggers):
        if user_data:
            expenses = user_data.get("expenses", [])
            if not expenses:
                return "You haven't recorded any expenses yet."
            else:
                last_expense = expenses[-1]
                desc = last_expense.get("description", "No description")
                amt = last_expense.get("total_amount", 0)
                return f"Your last expense was '{desc}' costing ₹{amt}."
        else:
            return "I couldn't fetch your expenses. Please try again."

    weekly_wants_triggers = ["weekly wants", "weekly spending", "wants this week", "how much spent this week", "weekly limit"]
    if any(trigger in msg for trigger in weekly_wants_triggers):
        if user_data:
            budget_limits = user_data.get("budget_limits", {})
            weekly_limit = budget_limits.get("Wants", 0)
            expenses = user_data.get("expenses", [])
            start_week = datetime.now() - timedelta(days=datetime.now().weekday())
            weekly_wants_total = sum(e.get("total_amount", 0) for e in expenses 
                                     if e.get("type") == "Wants" and to_naive(e.get("date")) >= start_week)
            return f"You have spent ₹{weekly_wants_total} on 'Wants' this week out of a limit of ₹{weekly_limit}."
        else:
            return "I couldn't fetch your weekly wants info. Please try again."

    funds_triggers = ["my funds", "available funds", "balance", "money i have", "account balance"]
    if any(trigger in msg for trigger in funds_triggers):
        if user_data:
            funds = user_data.get("available_funds", 0)
            return f"Your available funds are ₹{funds}."
        else:
            return "I couldn't fetch your funds info. Please try again."

    goals_triggers = ["goal", "goals", "what are my goals", "set goal", "financial goal"]
    if any(trigger in msg for trigger in goals_triggers):
        if user_data:
            goals = user_data.get("goals", [])
            zen_savings = user_data.get("zen_savings", 0)
            if not goals:
                return "You haven't set any financial goals yet."
            else:
                response_str = "Here are your active goals:\n"
                for goal in goals:
                    name = goal.get("name", "Unnamed goal")
                    target = goal.get("amount", 0)
                    remaining = max(0, target - zen_savings)
                    response_str += f"- {name}: Target ₹{target} (₹{remaining} left)\n"
                return response_str
        else:
            return "I couldn't fetch your goals. Please try again."

    setup_limit_triggers = ["setup weekly limit", "set weekly limit", "change weekly limit", "update weekly limit"]
    if any(trigger in msg for trigger in setup_limit_triggers):
        return "You can set or update your weekly 'Wants' limit on the Add Expense page by entering your desired limit."

    expense_general_triggers = ["expense", "spend", "cost", "money", "record expense", "track expense"]
    if any(trigger in msg for trigger in expense_general_triggers):
        return "Managing your expenses is crucial. You can add a new expense on the Add Expense page. Let me know if you need help with that."

    if "thank" in msg:
        return "You're welcome! I'm here to help."

    return "I'm sorry, I didn't quite understand that. Could you please rephrase or provide more details?"

def show_add_expense():
    st.title("➕ Add New Expense")
    user_data = get_user_data()
    if not user_data:
        st.error("User data not found.")
        return
    available_funds = user_data.get("available_funds", 0)
    budget_limits = user_data.get("budget_limits", {})
    wants_limit = budget_limits.get("Wants", 0)
    zen_mode = user_data.get("zen_mode", False)
    zen_savings = user_data.get("zen_savings", 0)
    expenses = user_data.get("expenses", [])
    
    st.write(f"💰 Available Funds: ₹{available_funds}")
    st.write(f"🎯 Weekly 'Wants' Limit: ₹{wants_limit}")
    add_to_limit = st.number_input("Add more to your weekly 'Wants' limit", min_value=0, step=100, key="add_limit")
    if st.button("➕ Increase Limit"):
        wants_limit += add_to_limit
        budget_limits["Wants"] = wants_limit
        user_data["budget_limits"] = budget_limits
        update_user_data(user_data)
        st.success(f"Weekly 'Wants' limit increased to ₹{wants_limit}!")
        safe_rerun()
    
    start_week = datetime.now() - timedelta(days=datetime.now().weekday())
    weekly_wants_total = sum(e["total_amount"] for e in expenses 
                             if e.get("type") == "Wants" and to_naive(e.get("date")) >= start_week)
    st.write(f"🧾 'Wants' Spent This Week: ₹{weekly_wants_total}")
    
    if "pending_expense" not in st.session_state:
        st.session_state.pending_expense = None
    if st.session_state.pending_expense:
        pending = st.session_state.pending_expense
        st.warning("🧘 Zen Mode: You've used more than 50% of your 'Wants' limit.")
        st.info("Do you really need this? Save now, thrive later. 🌱")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Add Expense Anyway"):
                expense_data = {
                    "description": pending["description"],
                    "total_amount": pending["amount"],
                    "category": pending["category"],
                    "type": pending["type"],
                    "date": datetime.now().isoformat()
                }
                expenses.append(expense_data)
                user_data["expenses"] = expenses
                user_data["available_funds"] = available_funds - pending["amount"]
                st.session_state.pending_expense = None
                update_user_data(user_data)
                st.success("✅ Expense added successfully!")
                safe_rerun()
        with col2:
            if st.button("❌ Cancel Expense"):
                saved_amount = pending["amount"]
                current_data = get_user_data()
                if current_data:
                    new_savings = current_data["zen_savings"] + saved_amount
                    update_user_data({
                        "zen_savings": new_savings,
                        "available_funds": current_data["available_funds"]
                    })
                    st.session_state.pending_expense = None
                    st.success(f"₹{saved_amount} saved as XP! 🎉")
                    check_pet_level_up()
                    safe_rerun()
    
    with st.form("expense_form"):
        description = st.text_input("Expense Description")
        amount = st.number_input("Amount", min_value=1, step=1)
        submit = st.form_submit_button("Add Expense")
        if submit:
            category_vector = vectorizer.transform([description])
            predicted_type = type_clf.predict(category_vector)[0]
            if predicted_type == "Wants":
                predicted_category = cat_clf.predict(category_vector)[0]
            else:
                vector_needs = vectorizer_needs.transform([description])
                predicted_category = needs_cat_clf.predict(vector_needs)[0]
            
            predicted_category = fallback_category(description, predicted_category)
            
            if available_funds <= 0:
                st.error("🚫 You have no available funds.")
                return
            if predicted_type == "Wants":
                if zen_mode and (weekly_wants_total + amount > 0.5 * wants_limit):
                    st.session_state.pending_expense = {
                        "description": description,
                        "amount": amount,
                        "category": predicted_category,
                        "type": predicted_type
                    }
                    st.warning("🧘 Zen Mode triggered: Spending check required.")
                    safe_rerun()
                    return
                if weekly_wants_total + amount > wants_limit:
                    st.error("🚫 This expense exceeds your weekly 'Wants' limit!")
                    return
            new_expense = {
                "description": description,
                "total_amount": amount,
                "category": predicted_category,
                "type": predicted_type,
                "date": datetime.now().isoformat()
            }
            expenses.append(new_expense)
            user_data["expenses"] = expenses
            user_data["available_funds"] = available_funds - amount
            update_user_data(user_data)
            st.success("✅ Expense added successfully!")
            safe_rerun()
    
    if wants_limit > 0:
        perc = weekly_wants_total / wants_limit
        if perc >= 1.0:
            st.error("🚨 You've hit your weekly 'Wants' limit!")
        elif perc >= 0.75:
            st.warning("⚠ You're at 75% of your 'Wants' limit.")
        elif perc >= 0.5:
            st.info("ℹ You're at 50% of your 'Wants' limit.")
        elif perc >= 0.25:
            st.info("ℹ You've reached 25% of your 'Wants' limit.")

def show_expense_history():
    st.title("📜 Expense History")
    user_data = get_user_data()
    today = datetime.now()
    start_week = today - timedelta(days=today.weekday())
    last_updated_str = user_data.get("wants_limit_last_updated")
    last_updated = datetime.fromisoformat(last_updated_str) if last_updated_str and isinstance(last_updated_str, str) else None
    if not last_updated or last_updated < start_week:
        user_data["weekly_wants_limit"] = 0
        user_data["wants_limit_last_updated"] = today.isoformat()
        update_user_data(user_data)

    if user_data and user_data.get("expenses"):
        # Create DataFrame from expenses
        df = pd.DataFrame(user_data["expenses"])

        # Convert date strings to datetime (all dates to naive using our helper)
        df['datetime'] = df['date'].apply(lambda d: to_naive(d))
        df['Date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['Time'] = df['datetime'].dt.strftime('%H:%M')

        df = df.rename(columns={
            'description': 'Description',
            'total_amount': 'Amount',
            'category': 'Category',
            'type': 'Type'
        })

        df = df.sort_values('datetime', ascending=False)

        df['Description'] = df['Description'].str.replace(r'^Added funds:.*', 'Added funds', regex=True)

        st.subheader("Cash Transactions")

        display_df = df[['Date', 'Time', 'Description', 'Category', 'Type', 'Amount']].rename(columns={'Amount': 'Amount (₹)'})

        def highlight_amount(row):
            color = 'green' if row['Description'] == 'Added funds' else 'red'
            styles = [''] * len(display_df.columns)
            amount_idx = display_df.columns.get_loc('Amount (₹)')
            styles[amount_idx] = f'color: {color}'
            return styles

        styled_df = (
            display_df
            .style
            .apply(highlight_amount, axis=1)
            .format({'Amount (₹)': '₹{:.0f}'})
        )

        st.write(styled_df)
        st.markdown("---")
    else:
        st.info("No expenses recorded yet.")

    st.subheader("💳 Paytm Transactions")
    paytm_data = []
    for i in range(10):
        description = random.choice(["Bought coffee", "Shopping on Amazon", "Uber ride", "Booked movie tickets", "Grocery shopping"])
        amount = random.randint(100, 5000)
        predicted_category = model.predict([description])[0]
        paytm_data.append({
            "description": description,
            "total_amount": amount,
            "category": predicted_category,
            "date": datetime.now().isoformat()
        })

    paytm_df = pd.DataFrame(paytm_data)
    paytm_df['Date'] = pd.to_datetime(paytm_df['date'], utc=True).dt.tz_convert(None).dt.strftime('%Y-%m-%d')
    paytm_df['Time'] = pd.to_datetime(paytm_df['date'], utc=True).dt.tz_convert(None).dt.strftime('%H:%M')
    paytm_df['Description'] = paytm_df['description']
    paytm_df['Category'] = paytm_df['category']
    paytm_df = paytm_df.rename(columns={'total_amount': 'Amount'})

    st.dataframe(paytm_df[['Date', 'Time', 'Description', 'Category', 'Amount']])
    total_paytm_spent = paytm_df['Amount'].sum()
    st.write(f"Total Spent via Paytm: ₹{total_paytm_spent}")

    available_funds = user_data.get("available_funds", 0) - total_paytm_spent
    user_data["available_funds"] = available_funds
    weekly_wants_total = sum(e.get("total_amount", 0) for e in user_data["expenses"]
                             if e.get("type") == "Wants" and to_naive(e.get("date")) >= start_week)
    if weekly_wants_total + total_paytm_spent > user_data["budget_limits"].get("Wants", 0):
        st.error("🚫 This expense exceeds your weekly 'Wants' limit!")
    else:
        st.success("Your weekly budget is updated!")
    update_user_data(user_data)

def show_funds_and_goals():
    st.title("💰 Funds & Goals")
    user_data = get_user_data()
    
    st.header("💵 Manage Funds")
    available_funds = user_data.get("available_funds", 0)
    st.write(f"Current Available Funds: ₹{available_funds}")
    
    with st.expander("➕ Add Funds"):
        with st.form("add_funds_form"):
            add_amount = st.number_input("Enter amount to add", min_value=1, step=1)
            submitted = st.form_submit_button("Add Funds")
            if submitted:
                available_funds += add_amount
                expense_data = {
                    "description": f"Added funds",
                    "total_amount": add_amount,
                    "category": "Miscellaneous",
                    "type": "Wants",
                    "date": datetime.now().isoformat()
                }
                user_data["expenses"].append(expense_data)
                update_user_data({
                    "available_funds": available_funds,
                    "expenses": user_data["expenses"]
                })
                st.success(f"₹{add_amount} added to your account!")
                safe_rerun()

    st.header("🎯 Financial Goals")
    goals = user_data.get("goals", [])
    zen_savings = user_data.get("zen_savings", 0)
    
    with st.expander("➕ Set New Goal"):
        with st.form("goal_form"):
            goal_name = st.text_input("Enter your goal (e.g., New Phone)")
            goal_amount = st.number_input("Amount required", min_value=1)
            submitted = st.form_submit_button("Add Goal")
            if submitted and goal_name:
                goals.append({"name": goal_name, "amount": goal_amount})
                update_user_data({"goals": goals})
                st.success("Goal added! 🎯")
    
    st.subheader("🧘 Zen Savings Progress")
    st.write(f"Total saved through Zen Mode: ₹{zen_savings} 💸")
    
    st.subheader("📌 Active Goals")
    if not goals:
        st.info("No goals set yet.")
    else:
        remaining_goals = []
        for goal in goals:
            goal_name = goal["name"]
            goal_amount = goal["amount"]
            if zen_savings >= goal_amount:
                st.success(f"🎉 Achieved: {goal_name} (₹{goal_amount}) 🏆")
            else:
                remaining = goal_amount - zen_savings
                progress = zen_savings / goal_amount
                st.metric(f"{goal_name}", f"₹{zen_savings}/₹{goal_amount}")
                st.progress(min(progress, 1.0))
                remaining_goals.append(goal)
        if remaining_goals != goals:
            update_user_data({"goals": remaining_goals})

def show_Zen():
    st.title("🧘 Zen")
    user_data = get_user_data()
    current_mode = user_data.get("zen_mode", False)
    st.write(f"Zen Mode is currently {'ON' if current_mode else 'OFF'}")
    if st.button("Toggle Zen Mode"):
        update_user_data({"zen_mode": not current_mode})
        st.success(f"Zen Mode turned {'ON' if not current_mode else 'OFF'}")
        safe_rerun()

def logout():
    st.session_state.current_user = None
    st.session_state.page = "Login"
    safe_rerun()

def show_login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.current_user = username
            st.session_state.page = "Home"
            st.success("Logged in successfully!")
            safe_rerun()
        else:
            st.error("Invalid credentials. Please try again.")
    if st.button("Go to Register"):
        st.session_state.page = "Register"
        safe_rerun()

def show_register():
    st.title("📝 Register")
    username = st.text_input("Choose a username", key="reg_username")
    password = st.text_input("Choose a password", type="password", key="reg_password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registration successful! Please login.")
            st.session_state.page = "Login"
            safe_rerun()
        else:
            st.error("Username already exists. Please choose another.")
    if st.button("Go to Login"):
        st.session_state.page = "Login"
        safe_rerun()

def show_home():
    st.title("🏠 Home")
    st.markdown("Welcome to FinZen, your intelligent finance companion! 🎯")
    
    user_data = get_user_data()
    
    tracking_enabled = st.checkbox(
        "Enable Financial Tracking", 
        value=user_data.get("tracking_enabled", True),
        key="tracking_toggle"
    )
    
    if tracking_enabled != user_data.get("tracking_enabled", True):
        update_user_data({"tracking_enabled": tracking_enabled})
        st.success("Tracking preference updated!")
        safe_rerun()
    
    if not tracking_enabled:
        st.warning("Financial tracking is currently disabled. Your expenses will not be recorded.")
        return
    
    if not user_data or not user_data.get("expenses"):
        st.info("No expenses recorded yet.")
        return
    
    df = pd.DataFrame(user_data["expenses"])
    
    # Convert dates to naive datetimes using our helper function.
    df['date'] = df['date'].apply(lambda d: to_naive(d))
    
    start_week = datetime.now() - timedelta(days=datetime.now().weekday())
    weekly_df = df[df['date'] >= start_week]
    
    if weekly_df.empty:
        st.info("No expenses recorded this week.")
        return
    
    st.subheader("📈 Weekly Spending Trend")
    daily_spending = weekly_df.groupby(weekly_df['date'].dt.date)['total_amount'].sum().reset_index()
    daily_spending.columns = ['Date', 'Total Amount']
    st.line_chart(daily_spending.set_index('Date'), use_container_width=True)
    
    st.subheader("📊 Spending by Category")
    chart_type = st.radio("Choose visualization:", ["Bar Chart", "Pie Chart"], horizontal=True)
    
    category_spending = weekly_df.groupby('category')['total_amount'].sum().reset_index()
    category_spending.columns = ['Category', 'Total Amount']
    
    if chart_type == "Bar Chart":
        chart = alt.Chart(category_spending).mark_bar().encode(
            x=alt.X('Category:O', axis=alt.Axis(labelAngle=0, title='Category')),
            y='Total Amount:Q',
            color='Category:N',
            tooltip=['Category', 'Total Amount']
        ).properties(
            width=alt.Step(40),
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        theme = st.get_option('theme.base')
        plt.style.use('dark_background' if theme == 'dark' else 'default')
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        wedges, texts, autotexts = ax.pie(
            category_spending['Total Amount'],
            labels=category_spending['Category'],
            autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
            startangle=90,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
        )
        
        for text in texts + autotexts:
            text.set_color('white' if theme == 'dark' else 'black')
            text.set_size(10)
        
        ax.axis('equal')
        ax.set_frame_on(False)
        st.pyplot(fig)
    
    st.subheader("💡 Personalized Savings Insights")
    total_spent = weekly_df['total_amount'].sum()
    if total_spent > 0:
        category_percentages = (category_spending['Total Amount'] / total_spent) * 100
        top_category_idx = category_spending['Total Amount'].idxmax()
        top_category = category_spending.loc[top_category_idx, 'Category']
        top_percent = category_percentages[top_category_idx]
        
        with st.expander(f"🔍 Your Top Spending Category: {top_category} ({top_percent:.1f}%)"):
            if top_category == "Food":
                st.markdown("""
                🥡 *Dining Optimization Tips:*
                - Meal prep 3x/week: Save ~₹1500 weekly
                - Use grocery lists: Reduce impulse buys by 30%
                - Bulk buying staples: Save 15% on essentials
                """)
            elif top_category == "Entertainment":
                st.markdown("""
                🎬 *Entertainment Savings:*
                - Host game nights: Save ₹500/outing
                - Library resources: Free books/movies
                - Early bird discounts: Save 20% on events
                """)
            elif top_category == "Shopping":
                st.markdown("""
                🛍 *Smart Shopping Strategies:*
                - 24-hour wait rule: Reduce impulse buys
                - Price tracking: Use Honey/Camel extensions
                - Seasonal sales: Plan purchases strategically
                """)
            else:
                st.markdown(f"""
                💡 *{top_category} Optimization:*
                - Review recurring subscriptions
                - Compare service providers
                - Bulk purchase discounts
                """)
    
    avg_daily = weekly_df.groupby(weekly_df['date'].dt.date)['total_amount'].sum().mean()
    with st.expander("📅 Weekly Spending Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Weekly Spend", f"₹{total_spent:.2f}")
        with col2:
            st.metric("Daily Average", f"₹{avg_daily:.2f}")
            
        if avg_daily > 1000:
            st.error("*High Spending Alert* 🚨")
            st.markdown(f"""
            Reduce daily spending by:
            - 10% = ₹{avg_daily*0.1:.0f}/day → ₹{avg_daily*0.1*30:.0f}/month savings
            - 20% = ₹{avg_daily*0.2:.0f}/day → ₹{avg_daily*0.2*30:.0f}/month savings
            """)
        else:
            st.success("*Spending Within Range* ✅")
            st.markdown("""
            Maintain good habits:
            - Track small purchases
            - Weekly budget reviews
            - Automated savings transfers
            """)
    
    wants_needs = weekly_df.groupby('type')['total_amount'].sum()
    with st.expander("⚖ Wants vs Needs Balance"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Essential Needs", f"₹{wants_needs.get('Needs', 0)}")
        with col2:
            st.metric("Discretionary Wants", f"₹{wants_needs.get('Wants', 0)}")
        
        if wants_needs.get('Wants', 0) > wants_needs.get('Needs', 1)*0.5:
            st.warning("*High Wants Spending*")
            st.markdown("""
            Balance suggestions:
            - 48-hour cooling period for wants
            - Allocate 30% wants budget to savings
            - Prioritize experience-based purchases
            """)
        else:
            st.success("*Healthy Balance*")
            st.markdown("""
            Good financial discipline!
            - Consider automating savings
            - Invest in skill development
            - Plan for long-term goals
            """)
    
    with st.expander("💰 Pro Savings Strategies"):
        st.markdown(f"""
        *Smart Money Moves:*
        🐾 *FinPet Benefits:* 
        - Level {user_data.get('pet_level', 1)} rewards: Better cashback offers
        - Current XP Savings: ₹{user_data.get('zen_savings', 0)}
        
        🔄 *Budget Hacks:*
        - Review weekly wants limit
        - Automate bill payments
        - Use cash envelopes for discretionary spending
        
        📱 *App Features:*
        - Zen Mode savings tracker
        - AI Chatbot for instant advice
        - Expense categorization insights
        """)
    
    st.subheader("⚡ Quick Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Funds", f"₹{user_data.get('available_funds', 0)}")
    with col2:
        st.metric("Weekly Wants Left", f"₹{user_data['budget_limits']['Wants'] - wants_needs.get('Wants', 0)}")
    with col3:
        st.metric("Zen Savings", f"₹{user_data.get('zen_savings', 0)}")

def main():
    if not st.session_state.current_user:
        if st.session_state.page == "Login":
            show_login()
        elif st.session_state.page == "Register":
            show_register()
        st.stop()
    st.sidebar.title(f"Welcome, {st.session_state.current_user} 👋")
    pages = [
        "Home", 
        "Add Expense", 
        "Funds & Goals",  
        "Expense History", 
        "FinPet", 
        "Weekly Wants",
        "Chatbot", 
        "Zen", 
        "Logout"
    ]
    st.session_state.page = st.sidebar.radio("Go to", pages)
    
    if st.session_state.page == "Home":
        show_home()
    elif st.session_state.page == "Add Expense":
        show_add_expense()
    elif st.session_state.page == "Funds & Goals":
        show_funds_and_goals()
    elif st.session_state.page == "Expense History":
        show_expense_history()
    elif st.session_state.page == "FinPet":
        show_finpet()
    elif st.session_state.page == "Weekly Wants":
        st.title("📅 Weekly 'Wants' Expense Tracker")
        user_data = get_user_data()
        expenses = user_data.get("expenses", [])
        budget_limits = user_data.get("budget_limits", {})
        weekly_limit = budget_limits.get("Wants", 0)
        start_week = datetime.now() - timedelta(days=datetime.now().weekday())
        weekly_wants_total = sum(e["total_amount"] for e in expenses 
                                 if e.get("type") == "Wants" and to_naive(e.get("date")) >= start_week)
        st.write(f"### 🧾 You've spent ₹{weekly_wants_total} on 'Wants' this week.")
        st.write(f"🎯 Weekly Limit: ₹{weekly_limit}")
        if weekly_limit > 0:
            perc = weekly_wants_total / weekly_limit
            st.progress(min(perc, 1.0))
            if perc >= 1.0:
                st.error("🚨 You've hit your weekly 'Wants' limit!")
            elif perc >= 0.75:
                st.warning("⚠ You're at 75% of your 'Wants' limit.")
            elif perc >= 0.5:
                st.info("You're at 50% of your 'Wants' limit.")
            elif perc >= 0.25:
                st.info("You've reached 25% of your 'Wants' limit.")
        else:
            st.warning("⚠ You haven't set a weekly 'Wants' limit yet. Set it from the Add Expense page.")
    elif st.session_state.page == "Chatbot":
        show_chatbot()
    elif st.session_state.page == "Zen":
        show_Zen()
    elif st.session_state.page == "Logout":
        logout()

if __name__ == "__main__":
    main()
