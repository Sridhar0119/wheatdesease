import sqlite3

def check_db():
    conn = sqlite3.connect('wheat_disease.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("TABLES:", [t[0] for t in tables])
    for table_name in [t[0] for t in tables]:
        print("TABLE:", table_name)
        cursor.execute(f"PRAGMA table_info({table_name});")
        info = cursor.fetchall()
        for col in info:
            print("COL:", col)
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print("COUNT:", count)

    conn.close()

if __name__ == "__main__":
    check_db()
