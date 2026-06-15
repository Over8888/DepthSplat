import dayjs from 'dayjs';
import type { PresetScriptItem, TaskDetail, TaskHistoryFilters, TaskHistoryItem } from '@/types/api';
import { readJsonStorage, writeJsonStorage } from '@/utils/storage';

const STORAGE_KEY = 'depthsplat-task-history';

const VALID_STATES = new Set(['queued', 'preparing', 'running', 'postprocessing', 'success', 'failed', 'cancelled', 'missing']);

const isValidTaskHistoryItem = (item: unknown): item is TaskHistoryItem => {
  if (!item || typeof item !== 'object') return false;
  const record = item as Record<string, unknown>;
  return typeof record.id === 'string' && record.id.trim().length > 0 && typeof record.state === 'string' && VALID_STATES.has(record.state);
};

export const getTaskHistory = (): TaskHistoryItem[] => {
  const items = readJsonStorage<unknown[]>(STORAGE_KEY, []);
  const cleaned = items.filter(isValidTaskHistoryItem);
  if (cleaned.length !== items.length) {
    writeJsonStorage(STORAGE_KEY, cleaned);
  }
  return cleaned;
};

export const upsertTaskHistory = (task: Pick<TaskDetail, 'id' | 'sampleId' | 'sampleName' | 'presetId' | 'presetName' | 'state' | 'createdAt' | 'updatedAt'>) => {
  const current = getTaskHistory();
  const next = [
    {
      id: task.id,
      sampleId: task.sampleId,
      sampleName: task.sampleName,
      presetId: task.presetId,
      presetName: task.presetName,
      state: task.state,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    },
    ...current.filter((item) => item.id !== task.id),
  ].slice(0, 200);

  writeJsonStorage(STORAGE_KEY, next);
  return next;
};

export const markTaskHistoryMissing = (taskId: string) => {
  const current = getTaskHistory();
  const next = current.map((item) =>
    item.id === taskId
      ? {
          ...item,
          state: 'missing' as const,
          updatedAt: new Date().toISOString(),
        }
      : item,
  );

  writeJsonStorage(STORAGE_KEY, next);
  return next;
};

const normalizeText = (value?: string) => (value ?? '').trim().toLowerCase();

const matchesPresetFilter = (item: TaskHistoryItem, presetId: string, presets?: PresetScriptItem[]) => {
  if (item.presetId === presetId) return true;

  const preset = presets?.find((entry) => entry.id === presetId);
  if (!preset) return false;

  const presetName = normalizeText(preset.name);
  if (!presetName) return false;

  const itemPresetName = normalizeText(item.presetName);
  const itemSampleName = normalizeText(item.sampleName);

  return itemPresetName === presetName || itemSampleName.includes(presetName);
};

export const filterTaskHistory = (items: TaskHistoryItem[], filters: TaskHistoryFilters, presets?: PresetScriptItem[]) =>
  items.filter((item) => {
    if (!item.id?.trim()) return false;
    if (filters.state && filters.state !== 'all' && item.state !== filters.state) return false;
    if (filters.presetId && !matchesPresetFilter(item, filters.presetId, presets)) return false;
    if (filters.timeRange) {
      const [start, end] = filters.timeRange;
      const target = item.createdAt ?? item.updatedAt;
      if (!target) return false;
      const targetAt = dayjs(target);
      if (!targetAt.isValid()) return false;
      if (targetAt.isBefore(dayjs(start)) || targetAt.isAfter(dayjs(end))) return false;
    }
    return true;
  });
