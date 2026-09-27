import secrets
from datetime import datetime, timedelta, timezone


OTP_EXPIRE_MINUTES = 10


def generate_otp() -> str:
    """Generate a secure 4-digit OTP."""
    return f"{secrets.randbelow(10_000):04d}"


def get_otp_expiration() -> datetime:
    """Return the OTP expiration time."""
    return datetime.now(timezone.utc) + timedelta(
        minutes=OTP_EXPIRE_MINUTES
    )