import dayjs from 'dayjs';
import type { BackendTaskState, TaskDetail, TaskResult } from '@/types/api';
import { ACTIVE_TASK_STATES, CANCELLABLE_TASK_STATES, TERMINAL_TASK_STATES } from './constants';

export const isTaskActive = (state?: BackendTaskState) => !!state && ACTIVE_TASK_STATES.includes(state);
export const isTaskTerminal = (state?: BackendTaskState) => !!state && TERMINAL_TASK_STATES.includes(state);
export const isTaskCancellable = (state?: BackendTaskState) => !!state && CANCELLABLE_TASK_STATES.includes(state);
export const isSuccessTask = (state?: BackendTaskState) => state === 'success';
export const isFailedTask = (state?: BackendTaskState) => state === 'failed';
export const isCancelledTask = (state?: BackendTaskState) => state === 'cancelled';

export const formatTimestamp = (value?: string) => (value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '?');

export const formatDuration = (seconds?: number) => {
  if (seconds == null) return '?';
  if (seconds < 60) return `${seconds}\u79d2`;
  const minutes = Math.floor(seconds / 60);
  const remain = seconds % 60;
  return `${minutes}\u5206 ${remain}\u79d2`;
};

export const buildErrorSummary = (task?: TaskDetail) => {
  if (!task || task.state !== 'failed') return undefined;
  return task.error?.message ?? '\u4efb\u52a1\u5931\u8d25\uff0c\u8bf7\u67e5\u770b\u65e5\u5fd7\u4e86\u89e3\u66f4\u591a\u4fe1\u606f\u3002';
};

export const canShowPartialResults = (task?: TaskDetail, result?: TaskResult) => {
  if (!task || task.state !== 'cancelled') return false;
  return Boolean(result?.videoUrl || result?.depthImages?.length || result?.previewImages?.length);
};
