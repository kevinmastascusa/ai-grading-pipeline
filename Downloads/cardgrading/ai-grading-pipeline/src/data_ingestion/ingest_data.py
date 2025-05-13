import pandas as pd
from sqlalchemy.orm import Session
from .database_setup import get_db, Card # Assuming database_setup.py is in the same directory

def ingest_data_from_csv(csv_file_path: str, db: Session):
    """
    Reads card data from a CSV file and ingests it into the database.

    Args:
        csv_file_path (str): The path to the CSV file.
        db (Session): The SQLAlchemy database session.
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        for _, row in df.iterrows():
            # Ensure that 'name' is present, as it's a non-nullable field
            if 'name' not in row or pd.isna(row['name']):
                print(f"Skipping row due to missing name: {row}")
                continue

            card_data = {
                "name": row["name"],
                "grade": row.get("grade"),  # Use .get() for optional fields
                "image_path": row.get("image_path")
            }
            
            # Remove None values for optional fields if they are not provided in the CSV
            card_data = {k: v for k, v in card_data.items() if v is not None and not pd.isna(v)}

            card = Card(**card_data)
            db.add(card)
        
        db.commit()
        print(f"Successfully ingested data from {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
    except Exception as e:
        db.rollback()
        print(f"An error occurred during data ingestion: {e}")

if __name__ == "__main__":
    # This is an example of how to run the ingestion.
    # You'll need to create a sample CSV file or provide a path to an existing one.
    
    # Create a dummy CSV for demonstration if it doesn't exist
    sample_csv_path = "./data/sample_cards.csv"
    try:
        pd.DataFrame({
            'name': ['Card A', 'Card B', 'Card C', 'Card D'],
            'grade': ['9', '8', None, '10'],
            'image_path': ['/path/to/imageA.jpg', '/path/to/imageB.jpg', '/path/to/imageC.jpg', None]
        }).to_csv(sample_csv_path, index=False)
        print(f"Created sample CSV: {sample_csv_path}")
    except Exception as e:
        print(f"Could not create sample CSV: {e}")


    # Get a database session
    db_session_gen = get_db()
    db = next(db_session_gen)

    try:
        print(f"Attempting to ingest data from {sample_csv_path}...")
        ingest_data_from_csv(sample_csv_path, db)
        
        # Verify by querying data (optional)
        print("\nVerifying ingested data (first 5 entries):")
        all_cards = db.query(Card).limit(5).all()
        if all_cards:
            for card_entry in all_cards:
                print(f"ID: {card_entry.id}, Name: {card_entry.name}, Grade: {card_entry.grade}, Image Path: {card_entry.image_path}")
        else:
            print("No cards found in the database after ingestion attempt.")
            
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
    finally:
        db.close()
        print("\nIngestion process finished.")
