from sqlalchemy import text

from server.db.database import engine


try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Database connected successfully:", result.scalar())

except Exception as e:
    print("Database connection failed:")
    print(e)