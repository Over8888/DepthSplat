export type BackendTaskState =
  | 'queued'
  | 'preparing'
  | 'running'
  | 'postprocessing'
  | 'success'
  | 'failed'
  | 'cancelled'
  | 'missing';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

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
  sceneNumber?: string | number;
  inputViewCount?: number;
  targetViewCount?: number;
}

export interface PresetScriptItem {
  id: string;
  name: string;
  description?: string;
  checkpoint?: string;
  contextViews?: number;
  targetViews?: number;
  imageShape?: [number, number] | number[];
  sampleId?: string;
}

export interface TaskOptions {
  save_video: boolean;
  save_image: boolean;
  save_gt_image: boolean;
  save_input_images: boolean;
  compute_scores: boolean;
}

export interface CreateTaskFormData {
  sampleId?: string;
  preset?: string;
  inputViewCount?: number;
  video?: File;
  options?: TaskOptions;
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
  dataLoadSeconds?: number;
  dataPrepSeconds?: number;
  inferenceSeconds?: number;
  splatConversionSeconds?: number;
  renderSeconds?: number;
  postprocessSeconds?: number;
  scoreSeconds?: number;
  exportSeconds?: number;
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
  parameters?: Record<string, JsonValue | undefined>;
  cameraIntrinsics?: JsonValue;
  cameraExtrinsics?: JsonValue;
  inputImages?: string[];
  renderedImages?: string[];
  gtImages?: string[];
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

export interface TaskScores {
  psnr?: number;
  ssim?: number;
  lpips?: number;
  mean_pmr?: number;
  total_seconds?: number;
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
  splatUrl?: string;
  depthImages?: string[];
  renderedImages?: string[];
  gtImages?: string[];
  errorImages?: string[];
  comparisonConcatImage?: string;
  imagesZipUrl?: string;
  scores?: TaskScores;
  cameraParamsUrl?: string;
  previewImages?: string[];
  metrics?: TaskMetric[];
  parameters?: Record<string, JsonValue | undefined>;
  cameraIntrinsics?: JsonValue;
  cameraExtrinsics?: JsonValue;
  notes?: string;
}

export interface TaskHistoryItem {
  id: string;
  sampleId?: string;
  sampleName?: string;
  presetId?: string;
  presetName?: string;
  state: BackendTaskState;
  createdAt?: string;
  updatedAt?: string;
}

export interface TaskHistoryFilters {
  state?: BackendTaskState | 'all';
  presetId?: string;
  timeRange?: [string, string];
}

export interface InputImageCameraInfo {
  url: string;
  index: number;
  width?: number;
  height?: number;
  isContextView: boolean;
  participatesInInference: boolean;
  cameraIntrinsics?: { fx: number; fy: number; cx: number; cy: number };
  cameraExtrinsics?: { rotation: number[][]; translation: number[] };
  cameraParamsStatus: 'available' | 'missing' | 'partial';
}

export interface TaskInputImagesResponse {
  taskId: string;
  images: InputImageCameraInfo[];
}
