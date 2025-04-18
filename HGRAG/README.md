# Home Grown Retrieval Augmented Generation

This is my attempt to build a RAG system from scratch using nothing but Python and API access to Large Language Models.  My intent is to abstract out the LLM access as much as possible so that if we decide to swap it out, we can.  Right now I used OpenAI API emulation in LM Studio running small embedding and intelligence models.

## Architecture

- Utilize "text-embedding-nomic-embed-text-v1.5" for embedding
  - Creates a 768 dimensional vector for text
- Everything needs to be text
  - Needs to convert PDFs to Text (and images/image descriptions?)
  - Needs to convert Word documents to Text (and images and image descriptions)
  

- Create a text-based CSV file to store the chunks
  - How do we create a chunk of an appropriate size?  There are chunking libraries but we don't want to use those.
  - TODO: Look up chunking algorithms.

## Table Structure
### Documents
ID, FileName, FullText, DateTime

### Chunks
ID, DocID, ChunkNum, DocSect, ChunkText, Vector
