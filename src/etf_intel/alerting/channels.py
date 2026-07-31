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


class ResendEmailAlertChannel(AlertChannel):
    """Sends the alert as an email via the Resend HTTP API (no SMTP password).

    Mirrors the SPAI project's approach: POST to ``api.resend.com`` with a Bearer
    key. Requires a user-supplied ``RESEND_API_KEY`` and recipient.
    """

    _URL = "https://api.resend.com/emails"

    def __init__(
        self,
        api_key: str,
        recipient: str,
        sender: str = "ETF Intel <onboarding@resend.dev>",
    ):
        """Initialise with the Resend API key, recipient, and sender identity."""
        self.api_key = api_key
        self.recipient = recipient
        self.sender = sender

    def send(self, subject: str, body: str) -> None:
        """Send the alert as a simple HTML email via Resend."""
        import html as _html

        import requests

        html_body = (
            '<div style="font-family:Arial,sans-serif;max-width:640px;margin:auto;'
            'color:#222;">'
            f'<h2 style="color:#1a1a2e;">{_html.escape(subject)}</h2>'
            f'<pre style="font-family:inherit;font-size:14px;line-height:1.6;'
            f'white-space:pre-wrap;">{_html.escape(body)}</pre>'
            '<hr><p style="font-size:11px;color:#999;">ETF Intel — research signals, '
            "not financial advice.</p></div>"
        )
        resp = requests.post(
            self._URL,
            json={
                "from": self.sender,
                "to": [self.recipient],
                "subject": subject,
                "html": html_body,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Alert emailed to %s via Resend", self.recipient)


class TelegramAlertChannel(AlertChannel):
    """Sends the alert to a Telegram chat via the Bot API.

    Requires a bot token (from @BotFather) and a chat id. Never constructed
    automatically — the pipeline uses it only when both are set in the environment.
    """

    _URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str):
        """Initialise with the bot token and target chat id."""
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, subject: str, body: str) -> None:
        """Send the alert as a Telegram message (plain text, truncated to 4096)."""
        import requests

        text = f"{subject}\n\n{body}"[:4096]
        resp = requests.post(
            self._URL.format(token=self.bot_token),
            json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Alert sent to Telegram chat %s", self.chat_id)
