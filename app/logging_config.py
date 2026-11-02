from __future__ import annotations

import logging
import logging.config


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
                }
            },
            "filters": {"request_context": {"()": "app.logging_config.RequestContextFilter"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "filters": ["request_context"],
                }
            },
            "root": {"handlers": ["console"], "level": level.upper()},
        }
    )


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True
