
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_otp_email(to_email: str, otp: str) -> bool:
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", "587"))
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")

    if not all([host, username, password, email_from]):
        raise RuntimeError("Email configuration is missing in .env")

    message = EmailMessage()
    message["Subject"] = "JobComparisonAI - Verification Code"
    message["From"] = email_from
    message["To"] = to_email

    message.set_content(
        f"""Hello,

Your JobComparisonAI verification code is:

{otp}

This code is valid for a limited time.

If you did not request this code, please ignore this email.

Best regards,
JobComparisonAI
"""
    )

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)

    return True

