from src.utils.database import get_connection

conn = get_connection()

cursor = conn.cursor()

cursor.execute("SELECT version();")

print(cursor.fetchone())

cursor.close()
conn.close()