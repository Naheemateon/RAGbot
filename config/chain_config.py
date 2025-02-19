import dotenv
dotenv.load_dotenv()

# CONFIGURATION VARIABLES:
# model_name = 'llama3-8b-8192'
model_name = "gpt-4-0125-preview"
# model_name="llama3"
embeddings_model="text-embedding-ada-002"
temperature=0.1
chain_type = "map_reduce"
VECTOR_STORE_DIR = "vectordb"
chunk_size = 1024
chunk_overlap = 100
k = 3
score_threshold = 0.1