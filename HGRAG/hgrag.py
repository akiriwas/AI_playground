import argparse
import sys
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




def main():
    """
    Main function to parse command line arguments and run the application.
    """

    # Load the database
    try:
        conn = sqlite3.connect(database_fn)
        cursor = conn.cursor()
        logger.info("Successfully connected to {database_fn}")
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

    # 1. A simple positional argument (required).
    parser.add_argument("filename", help="The name of the file to process")

    # 2. An optional argument with a default value.
    parser.add_argument("-n", "--number", type=int, default=10, help="An integer number (default: 10)")

    # 3. A boolean flag (an option that is either present or absent).
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    # 4. An option with a limited set of choices
    parser.add_argument("-m", "--mode", choices=['a', 'b', 'c'], default='b', help="Choose a mode: a, b, or c (default: b)")

    # 5.  An argument that can take multiple values
    parser.add_argument("integers", metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')

    # Parse the command line arguments.  This converts the raw command line input
    # into a Python object (args) with attributes corresponding to the arguments.
    args = parser.parse_args()

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
        add_document(args.add)
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

   

    




if __name__ == "__main__":
    # This ensures that the main() function is called when the script is executed.
    # It's a common practice in Python to wrap the main part of your program
    # in a function and call it this way.  This helps with modularity and prevents
    # the code from running automatically if the script is imported as a module
    # into another script.
    main()

