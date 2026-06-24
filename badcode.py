import pickle

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id   # SQL injection
    return db.execute(query)

def divide(a, b):
    return a / b   # no zero-division guard

def load_config(raw):
    return pickle.loads(raw)   # unsafe deserialization

password = "admin123"   # hardcoded credential

def read_file(path):
    f = open(path)   # file never closed
    return f.read()
