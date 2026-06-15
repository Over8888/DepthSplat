from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Asia/Shanghai"


def app_timezone() -> ZoneInfo:
    return ZoneInfo(os.getenv("DEPTHSPLAT_V3_LOG_TIMEZONE", os.getenv("DEPTHSPLAT_V3_TASK_TIMEZONE", DEFAULT_TIMEZONE)))


def now_local() -> datetime:
    return datetime.now(app_timezone())


def to_local(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=app_timezone())
    return value.astimezone(app_timezone())
