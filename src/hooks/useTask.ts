import { useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { BackendTaskState, CreateTaskFormData } from '@/types/api';
import { taskApi } from '@/services/tasks';
import { upsertTaskHistory } from '@/store/taskHistoryStore';
import { isTaskActive, isTaskTerminal } from '@/utils/task';

export function useCreateTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: CreateTaskFormData) => taskApi.createTask(payload),
    onSuccess: (task) => {
      queryClient.invalidateQueries({ queryKey: ['task-history'] });
      upsertTaskHistory({ id: task.id, state: task.state, createdAt: task.createdAt, updatedAt: task.createdAt });
    },
  });
}

export function useTaskDetail(taskId?: string) {
  const query = useQuery({
    queryKey: ['task', taskId],
    queryFn: () => taskApi.getTask(taskId!),
    enabled: Boolean(taskId),
    refetchInterval: (queryInfo) => {
      const state = queryInfo.state.data?.state;
      return isTaskActive(state) ? 3000 : false;
    },
  });

  useEffect(() => {
    if (query.data) upsertTaskHistory(query.data);
  }, [query.data]);

  return query;
}

export function useTaskLogs(taskId?: string, taskState?: BackendTaskState, enabled = true) {
  return useQuery({
    queryKey: ['task-logs', taskId],
    queryFn: () => taskApi.getTaskLogs(taskId!),
    enabled: Boolean(taskId) && enabled,
    refetchInterval: () => {
      if (!taskState) return 2000;
      return isTaskTerminal(taskState) ? false : 2000;
    },
    refetchOnWindowFocus: false,
  });
}

export function useTaskResult(taskId?: string, enabled = true) {
  return useQuery({
    queryKey: ['task-result', taskId],
    queryFn: () => taskApi.getTaskResult(taskId!),
    enabled: Boolean(taskId) && enabled,
    retry: false,
  });
}

export function useCancelTask(taskId?: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => taskApi.cancelTask(taskId!),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['task', taskId] });
      await queryClient.invalidateQueries({ queryKey: ['task-logs', taskId] });
      await queryClient.invalidateQueries({ queryKey: ['task-result', taskId] });
    },
  });
}

export function useTaskInputImages(taskId?: string) {
  return useQuery({
    queryKey: ['task-input-images', taskId],
    queryFn: () => taskApi.getTaskInputImages(taskId!),
    enabled: Boolean(taskId),
    retry: false,
  });
}
