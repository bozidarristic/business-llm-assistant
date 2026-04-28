from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_backend: str = "transformers"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_max_new_tokens: int = 384

    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5

    chroma_collection_name: str = "business_assistant"
    chroma_persist_dir: str = "data/chroma"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
