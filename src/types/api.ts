export type BackendTaskState =
  | 'queued'
  | 'preparing'
  | 'running'
  | 'postprocessing'
  | 'success'
  | 'failed'
  | 'cancelled';

export type UiTaskState = BackendTaskState | 'submitting' | 'loading' | 'cancelling';

export interface SampleItem {
  id: string;
  name: string;
  description?: string;
  thumbnailUrl?: string;
  category?: string;
  tags?: string[];
  inputImages?: string[];
  previewImages?: string[];
}

export interface PresetScriptItem {
  id: string;
  name: string;
  description?: string;
  checkpoint?: string;
  contextViews?: number;
  imageShape?: [number, number] | number[];
  sampleId?: string;
}

export interface TaskOptions {
  testChunkInterval: boolean;
  saveVideo: boolean;
  computeScores: boolean;
  exportDepthMap: boolean;
}

export interface CreateTaskFormData {
  sampleId?: string;
  presetId?: string;
  images: File[];
  options: TaskOptions;
}

export interface CreateTaskResponse {
  id: string;
  state: BackendTaskState;
  createdAt?: string;
}

export interface TaskStageTiming {
  queuedAt?: string;
  startedAt?: string;
  finishedAt?: string;
  durationSeconds?: number;
  updatedAt?: string;
}

export interface TaskErrorSummary {
  message: string;
  code?: string;
  details?: string;
}

export interface TaskDetail {
  id: string;
  sampleId?: string;
  sampleName?: string;
  presetId?: string;
  presetName?: string;
  state: BackendTaskState;
  stage?: string;
  createdAt?: string;
  updatedAt?: string;
  submittedAt?: string;
  timings?: TaskStageTiming;
  parameters?: Record<string, string | number | boolean | null | undefined>;
  inputImages?: string[];
  error?: TaskErrorSummary;
}

export interface TaskLogEntry {
  id: string;
  timestamp?: string;
  level?: 'debug' | 'info' | 'warning' | 'error';
  message: string;
}

export interface TaskLogsResponse {
  taskId: string;
  entries: TaskLogEntry[];
}

export interface TaskMetric {
  label: string;
  value: string | number;
  unit?: string;
}

export interface TaskResult {
  taskId: string;
  state: BackendTaskState;
  inputImages?: string[];
  videoUrl?: string;
  depthImages?: string[];
  previewImages?: string[];
  metrics?: TaskMetric[];
  parameters?: Record<string, string | number | boolean | null | undefined>;
  notes?: string;
}

export interface TaskHistoryItem {
  id: string;
  sampleId?: string;
  sampleName?: string;
  state: BackendTaskState;
  createdAt?: string;
  updatedAt?: string;
}

export interface TaskHistoryFilters {
  state?: BackendTaskState | 'all';
  sampleId?: string;
  timeRange?: [string, string];
}
