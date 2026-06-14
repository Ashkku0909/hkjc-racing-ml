import os
import glob
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:your_password@127.0.0.1:5432/hkjc_db")

async def clean_database():
    print("Cleaning PostgreSQL database...")
    try:
        engine = create_async_engine(DB_URL, echo=False)
        async with engine.begin() as conn:
            # Drop tables if they exist
            await conn.execute(text("DROP TABLE IF EXISTS race_results CASCADE;"))
            await conn.execute(text("DROP TABLE IF EXISTS model_features CASCADE;"))
            await conn.execute(text("DROP TABLE IF EXISTS races CASCADE;"))
            await conn.execute(text("DROP TABLE IF EXISTS horses CASCADE;"))
        print("Database tables dropped successfully.")
    except Exception as e:
        print(f"Could not clean database (it might not be running or configured): {e}")

def clean_csvs():
    print("Cleaning CSV files...")
    
    # Clean raw CSVs
    raw_csv_dir = "data/raw_csvs"
    if os.path.exists(raw_csv_dir):
        files = glob.glob(os.path.join(raw_csv_dir, "*.csv"))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        print(f"Deleted {len(files)} raw CSV files from {raw_csv_dir}.")
    else:
        print(f"Directory {raw_csv_dir} does not exist.")
    
    # Clean model features CSV
    features_csv = "data/model_features.csv"
    if os.path.exists(features_csv):
        try:
            os.remove(features_csv)
            print(f"Deleted {features_csv}.")
        except Exception as e:
            print(f"Error deleting {features_csv}: {e}")
    else:
        print(f"File {features_csv} does not exist.")

async def main():
    print("--- Starting Data Cleanup ---")
    clean_csvs()
    await clean_database()
    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
