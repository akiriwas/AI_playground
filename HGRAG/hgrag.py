import argparse
import sys
import os
from datetime import datetime
import struct
import logging

import openai
import numpy as np

import sqlite3

import logging

# Get a logger instance (it's good practice to use a logger specific to your module)
logger = logging.getLogger(__name__)
# Set the default logging level (can be overridden later)
logger.setLevel(logging.WARNING)  # Or logging.INFO for more initial output
# Create a handler to direct log records (e.g., to the console)
handler = logging.StreamHandler()
# Create a formatter to define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set the formatter for the handler
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)


embedding_model = "text-embedding-nomic-embed-text-v1.5"
database_fn = "rag.db"

DEBUG = 0

def dprint(s):
    if DEBUG == 1:
        print(s)


def generate_embedding(text_to_AI):
    LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed
    
    # Set up the OpenAI-compatible client
    client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

    response = client.embeddings.create(
        model=global_model,
        input=text_to_AI
    )
    dprint(response)
    return response.data[0].embedding
    #return response["data"][0]["embedding"]

# Assumes that these are vector lists.
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def main():
    #Put the embeddings AND the appropriate text into a list
    vector_db = [(doc, generate_embedding(doc)) for doc in documents]

    # Query Example
    query = "Tell me about the Moon landing"
    query_embedding = generate_embedding(query)
    dprint(query_embedding)

    similarities = []
    for doc, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((doc, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("Query:", query)
    print("Top Matches:")
    ret = similarities[:2]
    for doc, similarity in similarities[:2]:
        print(f"- {doc} (score: {similarity})")


def check_db_empty(conn, cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # If there are no tables, the database is empty.
    ret = (len(tables) == 0)
    print("Database empty =", ret)
    return ret

# Returns True if format is correct
# Returns False otherwise
def check_db_format(conn, cursor):
    # Check for Documents table and its fields
    cursor.execute("PRAGMA table_info(Documents)")
    documents_fields = {row[1]: row[2] for row in cursor.fetchall()}  # {field_name: data_type}
    expected_documents_fields = {
        "ID": "INTEGER",
        "FileName": "TEXT",
        "FullText": "TEXT",
        "DateTime": "TEXT",
    }
    if documents_fields != expected_documents_fields:
        print("Documents table has incorrect fields or data types.")
        return False

    # Check for Chunks table and its fields
    cursor.execute("PRAGMA table_info(Chunks)")
    chunks_fields = {row[1]: row[2] for row in cursor.fetchall()}
    expected_chunks_fields = {
        "ID": "INTEGER",
        "DocID": "INTEGER",
        "ChunkNum": "INTEGER",
        "DocSect": "TEXT",
        "ChunkText": "TEXT",
        "Vector": "BLOB",
    }
    if chunks_fields != expected_chunks_fields:
        print("Chunks table has incorrect fields or data types.")
        return False

    print("Tables 'Documents' and 'Chunks' exist with the correct fields and data types.")
    return True


def main():
    """
    Main function to parse command line arguments and run the application.
    """

    # Load the database
    try:
        conn = sqlite3.connect(database_fn)
        cursor = conn.cursor()
        logger.info("Successfully connected to {database_fn}")

        if check_db_empty(conn, cursor):
            #DB is empty
            logger.info("Database is empty and tables are being created")
            # Empty database -- we just created it.  Fill it with blank tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Documents (
                    ID INTEGER PRIMARY KEY,
                    FileName TEXT,
                    FullText TEXT,
                    DateTime TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Chunks (
                    ID INTEGER PRIMARY KEY,
                    DocID INTEGER,
                    ChunkNum INTEGER,
                    DocSect TEXT,
                    ChunkText TEXT,
                    Vector BLOB,
                    FOREIGN KEY (DocID) REFERENCES Documents(ID)
                )
            """)
        elif not check_db_format(conn, cursor):
            #DB is NOT correctly formated
            logger.error("Database file is not correctly formated for HGRAG")
            return
        else:
            #DB is good to go and formated correctly
            pass

    except sqlite3.Error as e:
        logger.error("SQLite3 Error Occurred: {e}")
        return


    # Create an argument parser object.
    # The description provides a helpful message when the user runs the script with -h or --help.
    parser = argparse.ArgumentParser(description="This is a basic Python application that accepts command-line options.")

    parser.add_argument("-a", "--add", help="Add a file to the RAG database")
    parser.add_argument("-r", "--replace", help="Replace a file in the RAG database")
    parser.add_argument("-s", "--show", help="Show the database statistics")
    parser.add_argument("-q", "--query", help="Query the database")
    parser.add_argument("-sq", "--strict", help="Query the database STRICTLY using the information in the database")

    # Parse the command line arguments.  This converts the raw command line input
    # into a Python object (args) with attributes corresponding to the arguments.
    args = parser.parse_args()

    try:
        if args.show:
            # Show the statistics of the RAG database
            print("RAG Database Statistics")
            print("-----------------------")
            sql_query = "SELECT d.FileName, COUNT(c.ID) AS NumberOfChunks FROM Documents d JOIN Chunks c ON d.ID = c.DocID GROUP BY d.FileName;"
            cursor.execute(sql_query)
            docs = cursor.fetchall()
            for doc in docs:
                print(doc)
            return
        elif args.add:
            add_document(conn, cursor, args.add)
            return
        elif args.replace:
            replace_document(args.replace)
            return
        elif args.query:
            query(args.query, false)
            return
        elif args.strict:
            query(args.strict, true)
            return
        else:
            # Interactive chat
            print("Interactive Query")
            return
    except sqlite3.Error as e:
        logger.error("SQLite3 Error Occurred: {e}")
        return


def chunker_fixed(text, separator="\n\n", max_chunk_size=256, overlap=20):
    """
    Splits the input text using the separator, then chunks each block into
    segments of max_chunk_size with overlap.
    
    Parameters:
        text (str): The input text to be chunked.
        separator (str): String used to split the text into blocks.
        max_chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of characters that should overlap between chunks.
        
    Returns:
        List[str]: A list of chunked text segments.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")

    chunks = []
    blocks = text.split(separator)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        start = 0
        while start < len(block):
            end = min(start + max_chunk_size, len(block))
            chunk = block[start:end]
            chunks.append(chunk)
            if end == len(block):
                break
            start = end - overlap  # Slide the window with overlap

    return chunks

def read_file(filename):
    """
    Reads the entire content of a text file and returns it as a string.

    Args:
        filename (str): The name of the file to read.

    Returns:
        str: The text content of the file.  Returns an empty string
             if the file does not exist or an error occurs.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

def add_document(conn, cursor, doc_fn):
    #Filetypes supported = txt, pdf, doc
    file_extension = os.path.splitext(doc_fn)[1]
    print(file_extension)
    if file_extension not in ['.txt', '.md']:
        return

    text = read_file(doc_fn)
    if text is None:
        return false
    chunks = chunker_fixed(text, separator="\n\n", max_chunk_size=2048, overlap=300)
    print(chunks)

    # 1. Insert the document into the Documents table
    current_datetime = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO Documents (FileName, FullText, DateTime) VALUES (?, ?, ?)",
        (doc_fn, text, current_datetime),
    )
    doc_id = cursor.lastrowid  # Get the ID of the newly inserted document

    chunk_num = 0
    for chunk in chunks:
        vector = generate_embedding(chunk)
        vector_blob = struct.pack('f' * len(vector), *vector)
        cursor.execute(
            """
            INSERT INTO Chunks (DocID, ChunkNum, DocSect, ChunkText, Vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, chunk_num, 0, chunk, vector_blob)
        )
        chunk_num = chunk_num + 1
    conn.commit()

def replace_document(doc_fn):
    print("replace_document")

def query(query_text, strict):
    print("query")


def generate_embedding(text_to_AI):
    LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed
    
    # Set up the OpenAI-compatible client
    client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

    response = client.embeddings.create(
        model=embedding_model,
        input=text_to_AI
    )
    return response.data[0].embedding
   




if __name__ == "__main__":
    # This ensures that the main() function is called when the script is executed.
    # It's a common practice in Python to wrap the main part of your program
    # in a function and call it this way.  This helps with modularity and prevents
    # the code from running automatically if the script is imported as a module
    # into another script.
    main()

