import asyncio
import os
from pymongo import server_api
from motor.motor_asyncio import AsyncIOMotorClient  # dùng motor cho async Mongo
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

# Kết nối tới MongoDB (dùng biến môi trường)
MONGODB_URL = os.environ.get("MONGODB_URL")

# Tạo client bất đồng bộ (async)
client = AsyncIOMotorClient(
    MONGODB_URL,
    server_api=server_api.ServerApi(
        version="1",
        strict=True,
        deprecation_errors=True
    )
)

# Tạo database
db = client.get_database("EEG_Identification")

# Tạo collection
user_collection = db.get_collection("users")
eeg_collection = db.get_collection("eeg_records")
prediction_collection = db.get_collection("predictions")

# ✅ Ping test function
async def test_connection():
    try:
        result = await client.admin.command("ping")
        print("✅ MongoDB connection successful:", result)
    except Exception as e:
        print("❌ MongoDB connection failed:", e)


# Run test manually when running this file directly
if __name__ == "__main__":
    asyncio.run(test_connection())
