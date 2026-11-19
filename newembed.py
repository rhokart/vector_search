import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Load a model optimized for search and recommendation tasks
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def connect_to_mongodb():
    """Establishes connection to MongoDB."""
    client = MongoClient(
        'mongodb+srv://chandanbarikcb:NZ1iTUrOq7Dj8uuq@cluster0.rea57.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    return client, client.sample_mflix

def process_excel_data(file_path):
    """Reads and processes Excel file with correct column names."""
    columns = [
        'Category L1', 'Category L2', 'Category L3', 'Company Name / Brand Name',
        'Model Name', 'Conductor material', 'Length', 'Size', 'Color',
        'Listing unit (pieces)', 'HSN Code', 'GST Rate', 'Parent Listing', 'Child listing', 'SKU Listing',
        'Rated voltage', 'Product Image - 1', 'Product Image - 2', 'Product Image - 3',
        'Warranty', 'Product Description',
        'Product Description bullet-1', 'Product Description bullet-2', 'Product Description bullet-3', 'Product Description bullet-4',
        'Product Description bullet-5','About brand', 'Country of Origin',
        'Customer care address', 'Brand\'s contact number'
    ]

    try:
        df = pd.read_excel(file_path)
        df.columns = columns[:len(df.columns)]
        df = df.dropna(subset=['Product Description', 'SKU Listing'])  # Only keep rows with these fields
        df = df.where(pd.notna(df), None)  # Replace NaN with None
        return df
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        raise

def preprocess_text(text):
    """Preprocess text by removing special characters and standardizing format."""
    if text is None:
        return ""
    text = str(text)  # Ensure the input is a string
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def generate_description(record):
    """Generates a description from the record using only Product Description and SKU Listing."""
    description = f"{record.get('Product Description', '')} {record.get('SKU Listing', '')}"
    return description.strip()

def generate_embeddings(texts):
    """Generates embeddings for a batch of texts."""
    preprocessed_texts = [preprocess_text(text) for text in texts]
    embeddings = model.encode(preprocessed_texts, show_progress_bar=True)
    return embeddings.tolist()

def process_and_upload_batch(records, collection):
    """Processes and uploads a batch of records."""
    descriptions = [generate_description(record) for record in records]
    
    # Generate embeddings for this batch
    embeddings = generate_embeddings(descriptions)

    # Prepare records with embeddings
    for record, embedding in zip(records, embeddings):
        record['created_at'] = datetime.now()
        record['updated_at'] = datetime.now()
        record['embedding'] = embedding
    
    # Bulk insert into MongoDB
    collection.insert_many(records)

def upload_to_mongodb(db, df):
    """Splits data into chunks and uploads to MongoDB."""
    collection = db.ecommerce
    records = df.to_dict('records')
    batch_size = 500  # Process larger batches for efficiency
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_and_upload_batch, records[i:i + batch_size], collection)
            for i in range(0, len(records), batch_size)
        ]
        for future in as_completed(futures):
            future.result()

def main():
    """Main function to process and upload data."""
    file_path = input("Enter Excel filename (e.g., data.xlsx): ")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at: {file_path}")

    client, db = connect_to_mongodb()
    print("Processing Excel file...")
    df = process_excel_data(file_path)
    print("Uploading to MongoDB...")
    upload_to_mongodb(db, df)
    print("Upload complete!")
    client.close()

if __name__ == "__main__":
    main()