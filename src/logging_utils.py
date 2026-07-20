"""Defense-in-depth redaction for sensitive HTTP authentication headers."""
import logging
import re
from typing import Any

_SENSITIVE_HEADERS = re.compile(
    r"(?i)(authorization(?:['\"]?\s*[:=]\s*['\"]?\s*)bearer\s+|x-api-key['\"]?\s*[:=]\s*['\"]?\s*)([^\s,;'\"}\]]+)"
)


def redact_sensitive(value: Any) -> str:
    """Replace bearer and X-API-Key values without retaining them."""
    return _SENSITIVE_HEADERS.sub(r"\1[REDACTED]", str(value))


class SensitiveHeaderFilter(logging.Filter):
    """Redact sensitive values before a record reaches a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_sensitive(record.getMessage())
            record.args = ()
        except Exception:
            record.msg = "[log message redacted after formatting failure]"
            record.args = ()
        return True


def install_sensitive_header_filter() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(item, SensitiveHeaderFilter) for item in handler.filters):
            handler.addFilter(SensitiveHeaderFilter())
