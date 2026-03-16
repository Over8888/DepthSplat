import type {
  CreateTaskFormData,
  CreateTaskResponse,
  PresetScriptItem,
  SampleItem,
  TaskDetail,
  TaskLogsResponse,
  TaskResult,
} from '@/types/api';
import { request } from './http';

export const taskApi = {
  getSamples: async () => {
    const response = await request<{ items: SampleItem[] }>('/samples');
    return response.items ?? [];
  },

  getPresets: async () => {
    const response = await request<{ items: PresetScriptItem[] }>('/presets');
    return response.items ?? [];
  },

  createTask: async (payload: CreateTaskFormData) => {
    const formData = new FormData();

    if (payload.sampleId) formData.append('sampleId', payload.sampleId);
    if (payload.presetId) formData.append('presetId', payload.presetId);
    formData.append('options', JSON.stringify(payload.options));
    payload.images.forEach((file) => formData.append('images', file));

    return request<CreateTaskResponse>('/tasks', {
      method: 'POST',
      body: formData,
    });
  },

  getTask: (taskId: string) => request<TaskDetail>(`/tasks/${taskId}`),
  cancelTask: (taskId: string) => request<{ state: 'cancelled' | string }>(`/tasks/${taskId}/cancel`, { method: 'POST' }),
  getTaskLogs: (taskId: string) => request<TaskLogsResponse>(`/tasks/${taskId}/logs`),
  getTaskResult: (taskId: string) => request<TaskResult>(`/tasks/${taskId}/result`),
};
