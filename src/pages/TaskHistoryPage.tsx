import { Typography } from 'antd';
import { useNavigate } from 'react-router-dom';
import { TaskHistoryTable } from '@/components/TaskHistoryTable';
import { usePresets } from '@/hooks/useSamples';
import { useTaskHistory } from '@/hooks/useTaskHistory';

export function TaskHistoryPage() {
  const navigate = useNavigate();
  const { data: presets } = usePresets();
  const { items, filters, setFilters } = useTaskHistory(presets);

  return (
    <>
      <Typography.Title level={3}>{'\u4efb\u52a1\u5386\u53f2'}</Typography.Title>
      <TaskHistoryTable items={items} filters={filters} presets={presets} onFilterChange={setFilters} onOpen={(taskId) => navigate('/result', { state: { taskId } })} />
    </>
  );
}
