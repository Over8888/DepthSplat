import type { TaskDetail, TaskHistoryFilters, TaskHistoryItem } from '@/types/api';
import { readJsonStorage, writeJsonStorage } from '@/utils/storage';

const STORAGE_KEY = 'depthsplat-task-history';

export const getTaskHistory = (): TaskHistoryItem[] => readJsonStorage<TaskHistoryItem[]>(STORAGE_KEY, []);

export const upsertTaskHistory = (task: Pick<TaskDetail, 'id' | 'sampleId' | 'sampleName' | 'state' | 'createdAt' | 'updatedAt'>) => {
  const current = getTaskHistory();
  const next = [
    {
      id: task.id,
      sampleId: task.sampleId,
      sampleName: task.sampleName,
      state: task.state,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    },
    ...current.filter((item) => item.id !== task.id),
  ].slice(0, 200);

  writeJsonStorage(STORAGE_KEY, next);
  return next;
};

export const filterTaskHistory = (items: TaskHistoryItem[], filters: TaskHistoryFilters) =>
  items.filter((item) => {
    if (filters.state && filters.state !== 'all' && item.state !== filters.state) return false;
    if (filters.sampleId && item.sampleId !== filters.sampleId) return false;
    if (filters.timeRange) {
      const [start, end] = filters.timeRange;
      const created = item.createdAt ?? item.updatedAt;
      if (!created) return false;
      if (created < start || created > end) return false;
    }
    return true;
  });
