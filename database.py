from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

class Database:
    """
    Singleton wrapper for Supabase client.
    Ensures only one instance of the client is created and reused.
    """
    _instance: Client | None = None

    @classmethod
    def get_instance(cls) -> Client:
        if cls._instance is None:
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise ValueError("Supabase credentials are missing. Check your .env file.")
            cls._instance = create_client(SUPABASE_URL, SUPABASE_KEY)
        return cls._instance

def get_db() -> Client:
    """
    Returns the Supabase client instance.
    """
    return Database.get_instance()
