'''# src/data_ingestion/database_setup.py

from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the database URL (SQLite in this case)
# The database file will be created in the 'data' directory
DATABASE_URL = "sqlite:///./data/card_grading.db"

# Create a new SQLAlchemy engine instance
engine = create_engine(DATABASE_URL, echo=True) # echo=True for logging SQL statements

# Define a base for declarative models
Base = declarative_base()

# Define a sample table (e.g., for cards)
class Card(Base):
    __tablename__ = "cards"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    grade = Column(String, nullable=True) # Example: PSA 10, BGS 9.5
    image_path = Column(String, nullable=True) # Path to the card image

# Create a metadata instance
metadata = MetaData()

# Function to create database tables
def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

# Create a SessionLocal class to handle database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_db_tables()
'''
