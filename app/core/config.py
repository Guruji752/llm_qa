from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_token: str 
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chat_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
