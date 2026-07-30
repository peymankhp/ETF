"""Alert delivery channels.

Default channels (log/file) are side-effect free. Email and Telegram are provided
for convenience but require user-supplied credentials; they are never constructed
or invoked automatically by the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from etf_intel.common.logging import get_logger

logger = get_logger(__name__)


class AlertChannel(ABC):
    """Delivers an alert message somewhere."""

    @abstractmethod
    def send(self, subject: str, body: str) -> None:
        """Deliver ``body`` under ``subject``."""
        raise NotImplementedError


class LogAlertChannel(AlertChannel):
    """Writes the alert to the logger (default, safe)."""

    def send(self, subject: str, body: str) -> None:
        """Log the alert body line by line."""
        logger.info("ALERT: %s", subject)
        for line in body.splitlines():
            logger.info("  %s", line)


class FileAlertChannel(AlertChannel):
    """Appends the alert to a timestamped markdown file (safe)."""

    def __init__(self, directory: str | Path):
        """Initialise with the output directory."""
        self.directory = Path(directory)

    def send(self, subject: str, body: str) -> None:
        """Write the alert to ``<directory>/alert_<UTC-timestamp>.md``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.directory / f"alert_{stamp}.md"
        path.write_text(f"# {subject}\n\n{body}\n", encoding="utf-8")
        logger.info("Alert written -> %s", path)


class EmailAlertChannel(AlertChannel):
    """Sends the alert via SMTP. Requires user-supplied credentials.

    Not used automatically anywhere in the pipeline — construct it explicitly with
    your own SMTP settings if you want email delivery.
    """

    def __init__(self, host: str, port: int, username: str, password: str, recipient: str):
        """Initialise with SMTP credentials and the recipient address."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.recipient = recipient

    def send(self, subject: str, body: str) -> None:
        """Send the alert as a plain-text email."""
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = self.recipient
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port) as smtp:
            smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.send_message(msg)
        logger.info("Alert emailed to %s", self.recipient)
