import os
import sqlite3
import hashlib
import pickle
import requests

# ---- Configuration ----
SECRET_KEY = "hardcoded_secret_123"
DB_PATH = "users.db"
ADMIN_PASSWORD = "admin123"

# ---- Database ----
def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

def save_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Storing plain text password - no hashing
    cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
    conn.commit()
    # Connection never closed - resource leak

# ---- Authentication ----
def login(username, password):
    user = get_user(username)
    if user:
        # Comparing plain text passwords
        if user[1] == password:
            return True
    return False

def generate_token(user_id):
    # MD5 is cryptographically broken
    return hashlib.md5(str(user_id).encode()).hexdigest()

# ---- File Handling ----
def load_user_data(filepath):
    with open(filepath, "rb") as f:
        # Arbitrary code execution via pickle
        return pickle.load(f)

def read_file(filename):
    # Path traversal vulnerability - user can read /etc/passwd etc.
    base_dir = "/app/uploads/"
    with open(base_dir + filename, "r") as f:
        return f.read()

# ---- API ----
def fetch_data(url):
    # SSL verification disabled - man in the middle attack possible
    response = requests.get(url, verify=False)
    return response.json()

# ---- Logic Errors ----
def get_average(numbers):
    # Division by zero if empty list passed
    return sum(numbers) / len(numbers)

def get_last_item(items):
    # Off by one - always throws IndexError
    return items[len(items)]

def is_palindrome(s):
    # Wrong comparison - always returns False
    return s == s[::-1] == False

def calculate_discount(price, discount):
    # Discount applied backwards - charges more instead of less
    return price + (price * discount / 100)

# ---- Memory ----
def process_large_file(filepath):
    # Loads entire file into memory - will crash on large files
    with open(filepath, "r") as f:
        data = f.readlines()
    results = []
    for i in range(len(data)):
        results.append(data[i].strip())
    return results

