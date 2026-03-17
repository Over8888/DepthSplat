from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    queued = "queued"
    preparing = "preparing"
    running = "running"
    postprocessing = "postprocessing"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_STATES = {TaskState.success, TaskState.failed, TaskState.cancelled}


ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.queued: {TaskState.preparing, TaskState.cancelled, TaskState.failed},
    TaskState.preparing: {TaskState.running, TaskState.cancelled, TaskState.failed},
    TaskState.running: {TaskState.postprocessing, TaskState.cancelled, TaskState.failed},
    TaskState.postprocessing: {TaskState.success, TaskState.failed, TaskState.cancelled},
    TaskState.success: set(),
    TaskState.failed: set(),
    TaskState.cancelled: set(),
}
