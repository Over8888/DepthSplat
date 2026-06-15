import { useEffect, useMemo, useState } from 'react';
import { ApiError } from '@/services/http';
import { taskApi } from '@/services/tasks';
import type { PresetScriptItem, TaskHistoryFilters } from '@/types/api';
import { filterTaskHistory, getTaskHistory, markTaskHistoryMissing, upsertTaskHistory } from '@/store/taskHistoryStore';
import { isTaskTerminal } from '@/utils/task';

export function useTaskHistory(presets?: PresetScriptItem[]) {
  const [filters, setFilters] = useState<TaskHistoryFilters>({ state: 'all' });
  const [version, setVersion] = useState(0);
  const historyItems = useMemo(() => getTaskHistory(), [version]);
  const items = useMemo(() => filterTaskHistory(historyItems, filters, presets), [filters, historyItems, presets]);

  useEffect(() => {
    let cancelled = false;
    const syncTasks = async () => {
      const activeTaskIds = getTaskHistory()
        .filter((item) => typeof item.id === 'string' && item.id.trim().length > 0 && !isTaskTerminal(item.state))
        .map((item) => item.id);

      if (!activeTaskIds.length) return;

      const results = await Promise.allSettled(activeTaskIds.map((taskId) => taskApi.getTask(taskId)));
      if (cancelled) return;

      let changed = false;
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          upsertTaskHistory(result.value);
          changed = true;
        } else if (result.reason instanceof ApiError && result.reason.status === 404) {
          markTaskHistoryMissing(activeTaskIds[index]);
          changed = true;
        }
      });

      if (changed) setVersion((value) => value + 1);
    };

    void syncTasks();
    const timer = window.setInterval(syncTasks, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return {
    filters,
    items,
    setFilters,
    refresh: () => setVersion((value) => value + 1),
  };
}
