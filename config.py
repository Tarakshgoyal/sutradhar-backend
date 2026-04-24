from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "postgres")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "Shubhang07##"),
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "india-innovates")

SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "60"))

KAFKA_BATCH_TIMEOUT_SECONDS = int(os.getenv("KAFKA_BATCH_TIMEOUT_SECONDS", "60"))
KAFKA_BATCH_MAX_SIZE = int(os.getenv("KAFKA_BATCH_MAX_SIZE", "50"))

REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "60"))
REPORT_DATE_RANGE = os.getenv("REPORT_DATE_RANGE", "7d")

WEATHER_SCRAPE_INTERVAL_SECONDS = int(
    os.getenv("WEATHER_SCRAPE_INTERVAL_SECONDS", "21600")
)
WEATHER_HISTORICAL_BACKFILL_YEARS = int(
    os.getenv("WEATHER_HISTORICAL_BACKFILL_YEARS", "5")
)

SIGNAL_WORKER_INTERVAL_SECONDS = int(
    os.getenv("SIGNAL_WORKER_INTERVAL_SECONDS", "900")
)  # 15 min

# Auth / Email config
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "your_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_smtp_password")
SMTP_FROM = os.getenv("SMTP_FROM", "your_email@gmail.com")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "CHANGE_ME_to_a_very_long_random_secret_string_at_least_64_chars_long_1234567890abcdef",
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))

