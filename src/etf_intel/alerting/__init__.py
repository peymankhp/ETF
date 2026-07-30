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
    FileAlertChannel,
    LogAlertChannel,
)

__all__ = [
    "AlertChannel",
    "FileAlertChannel",
    "LogAlertChannel",
    "RankingChanges",
    "format_alert",
    "ranking_changes",
]
