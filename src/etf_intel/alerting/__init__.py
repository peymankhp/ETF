"""Alerting: detect week-over-week ranking changes and deliver a summary.

Delivery channels are pluggable. The default channels (log/file) are side-effect
free; email/Telegram are provided but require user-supplied credentials and are
never used automatically.
"""

from etf_intel.alerting.alerts import (
    RankingChanges,
    format_alert,
    ranking_changes,
)
from etf_intel.alerting.channels import (
    AlertChannel,
    EmailAlertChannel,
    FileAlertChannel,
    LogAlertChannel,
    ResendEmailAlertChannel,
    TelegramAlertChannel,
)

__all__ = [
    "AlertChannel",
    "EmailAlertChannel",
    "FileAlertChannel",
    "LogAlertChannel",
    "RankingChanges",
    "ResendEmailAlertChannel",
    "TelegramAlertChannel",
    "format_alert",
    "ranking_changes",
]
