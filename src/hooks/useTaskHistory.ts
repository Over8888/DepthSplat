import { useMemo, useState } from 'react';
import type { TaskHistoryFilters } from '@/types/api';
import { filterTaskHistory, getTaskHistory } from '@/store/taskHistoryStore';

export function useTaskHistory() {
  const [filters, setFilters] = useState<TaskHistoryFilters>({ state: 'all' });
  const [version, setVersion] = useState(0);
  const items = useMemo(() => filterTaskHistory(getTaskHistory(), filters), [filters, version]);

  return {
    filters,
    items,
    setFilters,
    refresh: () => setVersion((value) => value + 1),
  };
}
