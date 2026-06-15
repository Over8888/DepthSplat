import { useQuery } from '@tanstack/react-query';
import { taskApi } from '@/services/tasks';

export function useSamples(preset?: string) {
  return useQuery({
    queryKey: ['samples', preset],
    queryFn: () => taskApi.getSamples(preset),
    enabled: Boolean(preset),
  });
}

export function usePresets() {
  return useQuery({
    queryKey: ['presets'],
    queryFn: taskApi.getPresets,
  });
}
