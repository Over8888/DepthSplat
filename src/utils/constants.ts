import type { BackendTaskState, DepthSplatParameters } from '@/types/api';

export const TASK_STATE_LABELS: Record<BackendTaskState, string> = {
  queued: '\u6392\u961f\u4e2d',
  preparing: '\u51c6\u5907\u4e2d',
  running: '\u8fd0\u884c\u4e2d',
  postprocessing: '\u540e\u5904\u7406\u4e2d',
  success: '\u6210\u529f',
  failed: '\u5931\u8d25',
  cancelled: '\u5df2\u53d6\u6d88',
};

export const DEFAULT_PARAMETERS: DepthSplatParameters = {
  numInferenceSteps: 30,
  guidanceScale: 7,
  seed: undefined,
  outputFps: 24,
  exportDepthMap: true,
  outputFormat: 'mp4',
};

export const ACTIVE_TASK_STATES: BackendTaskState[] = ['queued', 'preparing', 'running', 'postprocessing'];
export const CANCELLABLE_TASK_STATES: BackendTaskState[] = ['preparing', 'running', 'postprocessing'];
export const TERMINAL_TASK_STATES: BackendTaskState[] = ['success', 'failed', 'cancelled'];
