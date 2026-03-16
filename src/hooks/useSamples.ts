import { useQuery } from '@tanstack/react-query';
import { taskApi } from '@/services/tasks';

export function useSamples() {
  return useQuery({
    queryKey: ['samples'],
    queryFn: taskApi.getSamples,
  });
}

export function usePresets() {
  return useQuery({
    queryKey: ['presets'],
    queryFn: taskApi.getPresets,
  });
}
