import os
from llama_index import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
    set_global_service_context
)
from llama_index.readers import PDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the Ollama LLM and local embedding model
ollama_llm = Ollama(model="mistral")
local_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a global ServiceContext
service_context = ServiceContext.from_defaults(
    llm=ollama_llm,
    embed_model=local_embed_model,
    chunk_size=300
)
set_global_service_context(service_context)

def get_index(data, index_name):
    index = None
    storage_context = StorageContext.from_defaults(persist_dir=index_name)
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(
            data,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist()
    else:
        print("Loading index", index_name)
        index = load_index_from_storage(storage_context)
    return index

# Load the PDF data
pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=pdf_path)

# Create or load the index
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()

# Example query
query = "What is the role of Canada in global trade?"
response = canada_engine.query(query)
print(response)
