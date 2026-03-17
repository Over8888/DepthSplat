from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        extra_fields = getattr(record, "fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, ensure_ascii=False)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]):
        event = kwargs.pop("event", None)
        fields = kwargs.pop("fields", None)
        extra = kwargs.get("extra", {})
        if event is not None:
            extra["event"] = event
        if fields is not None:
            extra["fields"] = fields
        kwargs["extra"] = extra
        return msg, kwargs


_DEF_HANDLER_NAME = "depthsplat_v3_stream"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "name", "") == _DEF_HANDLER_NAME:
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.name = _DEF_HANDLER_NAME
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)



def get_logger(name: str) -> StructuredLoggerAdapter:
    return StructuredLoggerAdapter(logging.getLogger(name), {})
