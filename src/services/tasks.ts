import type {
  BackendTaskState,
  CreateTaskFormData,
  CreateTaskResponse,
  InputImageCameraInfo,
  JsonValue,
  PresetScriptItem,
  SampleItem,
  TaskDetail,
  TaskInputImagesResponse,
  TaskLogsResponse,
  TaskResult,
  TaskScores,
  TaskStageTiming,
} from '@/types/api';
import { request, resolveBackendUrl } from './http';

type RawRecord = Record<string, unknown>;

const asRecord = (value: unknown): RawRecord => (value && typeof value === 'object' ? (value as RawRecord) : {});
const asString = (value: unknown) => (typeof value === 'string' ? value : undefined);
const asNumber = (value: unknown) => (typeof value === 'number' ? value : undefined);
const asDimensionNumber = (value: unknown) => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};
const asImageDimensionTuple = (value: unknown): [number | undefined, number | undefined] => {
  if (Array.isArray(value)) {
    return [asDimensionNumber(value[1] ?? value[0]), asDimensionNumber(value[0] ?? value[1])];
  }

  if (typeof value === 'string') {
    const match = value.match(/(\d+(?:\.\d+)?)\s*[x×,]\s*(\d+(?:\.\d+)?)/i);
    if (match) return [asDimensionNumber(match[1]), asDimensionNumber(match[2])];
  }

  return [undefined, undefined];
};
const asBoolean = (value: unknown) => (typeof value === 'boolean' ? value : undefined);
const asJsonValue = (value: unknown): JsonValue | undefined => {
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) {
    const items = value.map(asJsonValue).filter((item): item is JsonValue => item !== undefined);
    return items.length === value.length ? items : undefined;
  }
  if (value && typeof value === 'object') {
    const result: Record<string, JsonValue> = {};
    Object.entries(value as RawRecord).forEach(([key, entry]) => {
      const normalized = asJsonValue(entry);
      if (normalized !== undefined) result[key] = normalized;
    });
    return result;
  }
  return undefined;
};
const asStringArray = (value: unknown): string[] | undefined =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : undefined;
const asUrl = (value: unknown) => resolveBackendUrl(asString(value));
const asUrlArray = (value: unknown) => asStringArray(value)?.map((url) => resolveBackendUrl(url) ?? url);
const asPrimitiveRecord = (value: unknown): Record<string, JsonValue | undefined> | undefined => {
  const raw = asRecord(value);
  if (!Object.keys(raw).length) return undefined;

  const result: Record<string, JsonValue | undefined> = {};
  Object.entries(raw).forEach(([key, entry]) => {
    const normalized = asJsonValue(entry);
    if (normalized !== undefined || entry === undefined) {
      result[key] = normalized;
    }
  });

  return result;
};

const normalizeState = (value: unknown): BackendTaskState => {
  const state = typeof value === 'string' ? value : 'queued';
  if (state === 'queued' || state === 'preparing' || state === 'running' || state === 'postprocessing' || state === 'success' || state === 'failed' || state === 'cancelled' || state === 'missing') {
    return state;
  }
  return 'queued';
};

const normalizeTiming = (value: unknown): TaskStageTiming | undefined => {
  const raw = asRecord(value);
  if (!Object.keys(raw).length) return undefined;

  return {
    queuedAt: asString(raw.queuedAt ?? raw.queued_at),
    startedAt: asString(raw.startedAt ?? raw.started_at ?? raw.runningStartedAt ?? raw.running_started_at),
    finishedAt: asString(raw.finishedAt ?? raw.finished_at),
    durationSeconds: asNumber(raw.durationSeconds ?? raw.duration_seconds ?? raw.total_seconds),
    updatedAt: asString(raw.updatedAt ?? raw.updated_at ?? raw.finished_at),
    dataLoadSeconds: asNumber(raw.dataLoadSeconds ?? raw.data_load_seconds),
    dataPrepSeconds: asNumber(raw.dataPrepSeconds ?? raw.data_prep_seconds),
    inferenceSeconds: asNumber(raw.inferenceSeconds ?? raw.inference_seconds),
    splatConversionSeconds: asNumber(raw.splatConversionSeconds ?? raw.splat_conversion_seconds),
    renderSeconds: asNumber(raw.renderSeconds ?? raw.render_seconds),
    postprocessSeconds: asNumber(raw.postprocessSeconds ?? raw.postprocess_seconds ?? raw.postProcessingSeconds ?? raw.post_processing_seconds),
    scoreSeconds: asNumber(raw.scoreSeconds ?? raw.score_seconds ?? raw.computeScoresSeconds ?? raw.compute_scores_seconds),
    exportSeconds: asNumber(raw.exportSeconds ?? raw.export_seconds ?? raw.saveVideoSeconds ?? raw.save_video_seconds),
  };
};

const normalizeSample = (raw: unknown): SampleItem => {
  const item = asRecord(raw);
  return {
    id: asString(item.id ?? item.sample_id) ?? '',
    name: asString(item.name ?? item.sample_name ?? item.id ?? item.sample_id) ?? '',
    description: asString(item.description),
    thumbnailUrl: asUrl(item.thumbnailUrl ?? item.thumbnail_url),
    category: asString(item.category),
    tags: Array.isArray(item.tags) ? item.tags.filter((tag): tag is string => typeof tag === 'string') : undefined,
    inputImages: asUrlArray(item.inputImages ?? item.input_images ?? item.images),
    previewImages: asUrlArray(item.previewImages ?? item.preview_images),
    sceneNumber: asString(item.sceneNumber ?? item.scene_number ?? item.sceneId ?? item.scene_id) ?? asNumber(item.sceneNumber ?? item.scene_number ?? item.sceneId ?? item.scene_id),
    inputViewCount: asNumber(item.inputViewCount ?? item.input_view_count ?? item.inputViews ?? item.input_views ?? item.contextViews ?? item.context_views),
    targetViewCount: asNumber(item.targetViewCount ?? item.target_view_count ?? item.targetViews ?? item.target_views),
  };
};

const normalizePreset = (raw: unknown): PresetScriptItem => {
  const item = asRecord(raw);
  const imageShape = item.imageShape ?? item.image_shape;
  return {
    id: asString(item.id ?? item.preset_id) ?? '',
    name: asString(item.name ?? item.preset_name ?? item.id ?? item.preset_id) ?? '',
    description: asString(item.description),
    checkpoint: asString(item.checkpoint ?? item.checkpoint_path ?? item.checkpoint_name),
    contextViews: asNumber(item.contextViews ?? item.context_views ?? item.num_context_views),
    targetViews: asNumber(item.targetViews ?? item.target_views ?? item.num_target_views),
    imageShape: Array.isArray(imageShape) ? imageShape.filter((x): x is number => typeof x === 'number') : undefined,
    sampleId: asString(item.sampleId ?? item.sample_id),
  };
};

const normalizeCreateTaskResponse = (raw: unknown): CreateTaskResponse => {
  const item = asRecord(raw);
  return {
    id: asString(item.id ?? item.task_id) ?? '',
    state: normalizeState(item.state ?? item.status),
    createdAt: asString(item.createdAt ?? item.created_at),
  };
};

const normalizeTaskDetail = (raw: unknown): TaskDetail => {
  const item = asRecord(raw);
  const errorRaw = item.error ?? item.error_summary;
  const errorRecord = asRecord(errorRaw);

  return {
    id: asString(item.id ?? item.task_id) ?? '',
    sampleId: asString(item.sampleId ?? item.sample_id),
    sampleName: asString(item.sampleName ?? item.sample_name),
    presetId: asString(item.presetId ?? item.preset_id),
    presetName: asString(item.presetName ?? item.preset_name),
    state: normalizeState(item.state ?? item.status),
    stage: asString(item.stage),
    createdAt: asString(item.createdAt ?? item.created_at ?? item.submittedAt ?? item.submitted_at),
    updatedAt: asString(item.updatedAt ?? item.updated_at ?? item.finished_at),
    submittedAt: asString(item.submittedAt ?? item.submitted_at ?? item.created_at),
    timings: normalizeTiming(item.timings ?? item.timing),
    parameters: asPrimitiveRecord(item.parameters ?? item.options),
    cameraIntrinsics: asJsonValue(item.cameraIntrinsics ?? item.camera_intrinsics ?? item.intrinsics ?? item.K ?? item.k),
    cameraExtrinsics: asJsonValue(item.cameraExtrinsics ?? item.camera_extrinsics ?? item.extrinsics ?? item.cameraToWorld ?? item.camera_to_world ?? item.c2w),
    inputImages: asUrlArray(item.inputImages ?? item.input_images ?? item.images),
    renderedImages: asUrlArray(item.renderedImages ?? item.rendered_images ?? item.renderImages ?? item.render_images ?? item.predImages ?? item.pred_images),
    gtImages: asUrlArray(item.gtImages ?? item.gt_images ?? item.groundTruthImages ?? item.ground_truth_images ?? item.targetImages ?? item.target_images),
    error: Object.keys(errorRecord).length
      ? {
          message: asString(errorRecord.message ?? errorRecord.summary ?? errorRecord.error) ?? '????',
          code: asString(errorRecord.code),
          details: asString(errorRecord.details ?? errorRecord.detail),
        }
      : undefined,
  };
};

const normalizeScores = (raw: unknown): TaskScores | undefined => {
  if (!raw || typeof raw !== 'object') return undefined;
  const obj = raw as Record<string, unknown>;
  const scores: TaskScores = {};
  if (typeof obj.psnr === 'number') scores.psnr = obj.psnr;
  if (typeof obj.ssim === 'number') scores.ssim = obj.ssim;
  if (typeof obj.lpips === 'number') scores.lpips = obj.lpips;
  if (typeof obj.mean_pmr === 'number') scores.mean_pmr = obj.mean_pmr;
  if (typeof obj.total_seconds === 'number') scores.total_seconds = obj.total_seconds;
  return Object.keys(scores).length > 0 ? scores : undefined;
};

const normalizeTaskResult = (raw: unknown): TaskResult => {
  const item = asRecord(raw);
  return {
    taskId: asString(item.taskId ?? item.task_id ?? item.id) ?? '',
    state: normalizeState(item.state ?? item.status),
    inputImages: asUrlArray(item.inputImages ?? item.input_images ?? item.images),
    videoUrl: asUrl(item.videoUrl ?? item.video_url ?? item.output_video_url ?? item.result_video_url ?? item.video),
    splatUrl: asUrl(item.splatUrl ?? item.splat_url),
    depthImages: asUrlArray(item.depthImages ?? item.depth_images),
    renderedImages: asUrlArray(item.renderedImages ?? item.rendered_images ?? item.renderImages ?? item.render_images ?? item.predImages ?? item.pred_images),
    gtImages: asUrlArray(item.gtImages ?? item.gt_images ?? item.groundTruthImages ?? item.ground_truth_images ?? item.targetImages ?? item.target_images),
    errorImages: asUrlArray(item.errorImages ?? item.error_images),
    comparisonConcatImage: asUrl(item.comparisonConcatImage ?? item.comparison_concat_image),
    imagesZipUrl: asUrl(item.imagesZipUrl ?? item.images_zip_url),
    scores: normalizeScores(item.scores),
    cameraParamsUrl: asUrl(item.cameraParamsUrl ?? item.camera_params_url),
    previewImages: asUrlArray(item.previewImages ?? item.preview_images),
    metrics: Array.isArray(item.metrics) ? (item.metrics as TaskResult['metrics']) : undefined,
    parameters: asPrimitiveRecord(item.parameters ?? item.options),
    cameraIntrinsics: asJsonValue(item.cameraIntrinsics ?? item.camera_intrinsics ?? item.intrinsics ?? item.K ?? item.k),
    cameraExtrinsics: asJsonValue(item.cameraExtrinsics ?? item.camera_extrinsics ?? item.extrinsics ?? item.cameraToWorld ?? item.camera_to_world ?? item.c2w),
    notes: asString(item.notes ?? item.note ?? item.message),
  };
};

export const taskApi = {
  getSamples: async (preset?: string) => {
    const response = await request<{ items?: unknown[] } | unknown[]>('/samples', {
      query: {
        preset,
      },
    });
    const items = Array.isArray(response) ? response : response.items ?? [];
    return items.map(normalizeSample);
  },

  getPresets: async () => {
    const response = await request<{ items?: unknown[] } | unknown[]>('/presets');
    const items = Array.isArray(response) ? response : response.items ?? [];
    return items.map(normalizePreset);
  },

  createTask: async (payload: CreateTaskFormData) => {
    const formData = new FormData();

    if (payload.sampleId) formData.append('sampleId', payload.sampleId);
    if (payload.preset) formData.append('preset', payload.preset);
    if (payload.inputViewCount) formData.append('inputViewCount', String(payload.inputViewCount));
    if (payload.options) formData.append('options', JSON.stringify(payload.options));
    if (payload.video) formData.append('video', payload.video);

    const response = await request<unknown>('/tasks', {
      method: 'POST',
      body: formData,
    });

    return normalizeCreateTaskResponse(response);
  },

  getTask: async (taskId: string) => {
    const response = await request<unknown>(`/tasks/${taskId}`);
    return normalizeTaskDetail(response);
  },

  cancelTask: async (taskId: string) => {
    const response = await request<unknown>(`/tasks/${taskId}/cancel`, { method: 'POST' });
    const item = asRecord(response);
    return { state: String(item.state ?? item.status ?? 'cancelled') };
  },

  getTaskLogs: async (taskId: string) => {
    const response = await request<unknown>(`/tasks/${taskId}/logs`);
    const item = asRecord(response);
    const data = asRecord(item.data);
    const entries = Array.isArray(response)
      ? response
      : Array.isArray(item.entries)
        ? item.entries
        : Array.isArray(item.logs)
          ? item.logs
          : Array.isArray(data.entries)
            ? data.entries
            : Array.isArray(data.logs)
              ? data.logs
              : typeof response === 'string'
                ? response
                    .split(/\r?\n/)
                    .map((line) => line.trim())
                    .filter(Boolean)
                    .map((line, index) => ({ id: String(index), message: line }))
                : [];

    return {
      taskId: asString(item.taskId ?? item.task_id ?? data.taskId ?? data.task_id) ?? taskId,
      entries: entries.map((entry, index) => {
        const row = asRecord(entry);
        return {
          id: asString(row.id ?? row.logId ?? row.log_id) ?? String(index),
          timestamp: asString(row.timestamp ?? row.time ?? row.ts ?? row.createdAt ?? row.created_at),
          level: (asString(row.level ?? row.logLevel ?? row.log_level ?? row.severity) as 'debug' | 'info' | 'warning' | 'error' | undefined) ?? 'info',
          message: asString(row.message ?? row.text ?? row.log ?? row.content ?? row.line) ?? '',
        };
      }),
    } as TaskLogsResponse;
  },

  getTaskResult: async (taskId: string) => {
    const response = await request<unknown>(`/tasks/${taskId}/result`);
    return normalizeTaskResult(response);
  },

  getTaskInputImages: async (taskId: string) => {
    const response = await request<unknown>(`/tasks/${taskId}/input-images`);
    const item = asRecord(response);
    const images = Array.isArray(item.images) ? item.images : Array.isArray(response) ? response : [];
    return {
      taskId: asString(item.taskId ?? item.task_id) ?? taskId,
      images: images.map((img, idx) => {
        const i = asRecord(img);
        const intrinsics = asRecord(i.cameraIntrinsics ?? i.camera_intrinsics);
        const extrinsics = asRecord(i.cameraExtrinsics ?? i.camera_extrinsics);
        const resolution = i.resolution ?? i.imageResolution ?? i.image_resolution ?? i.size ?? i.shape ?? i.imageShape ?? i.image_shape;
        const resolutionRecord = asRecord(resolution);
        const [tupleWidth, tupleHeight] = asImageDimensionTuple(resolution);
        const width = asDimensionNumber(
          i.width ??
            i.w ??
            i.imageWidth ??
            i.image_width ??
            resolutionRecord.width ??
            resolutionRecord.w ??
            tupleWidth,
        );
        const height = asDimensionNumber(
          i.height ??
            i.h ??
            i.imageHeight ??
            i.image_height ??
            resolutionRecord.height ??
            resolutionRecord.h ??
            tupleHeight,
        );
        return {
          url: asUrl(i.url) ?? '',
          index: asNumber(i.index ?? idx) ?? idx,
          width,
          height,
          isContextView: asBoolean(i.isContextView ?? i.is_context_view) ?? false,
          participatesInInference: asBoolean(i.participatesInInference ?? i.participates_in_inference) ?? true,
          cameraIntrinsics: Object.keys(intrinsics).length ? {
            fx: (asNumber(intrinsics.fx) ?? asNumber(intrinsics.focal_x))!,
            fy: (asNumber(intrinsics.fy) ?? asNumber(intrinsics.focal_y))!,
            cx: (asNumber(intrinsics.cx) ?? asNumber(intrinsics.principal_x))!,
            cy: (asNumber(intrinsics.cy) ?? asNumber(intrinsics.principal_y))!,
          } : undefined,
          cameraExtrinsics: Object.keys(extrinsics).length ? {
            rotation: (Array.isArray(extrinsics.rotation) ? extrinsics.rotation : Array.isArray(extrinsics.R) ? extrinsics.R : []) as number[][],
            translation: (Array.isArray(extrinsics.translation) ? extrinsics.translation : Array.isArray(extrinsics.t) ? extrinsics.t : []) as number[],
          } : undefined,
          cameraParamsStatus: (asString(i.cameraParamsStatus ?? i.camera_params_status) as InputImageCameraInfo['cameraParamsStatus']) ?? (i.cameraIntrinsics || i.camera_extrinsics || i.cameraExtrinsics || i.camera_intrinsics ? 'available' : 'missing'),
        } as InputImageCameraInfo;
      }),
    } as TaskInputImagesResponse;
  },
};
