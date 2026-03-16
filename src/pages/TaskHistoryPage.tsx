import { Typography } from 'antd';
import { useNavigate } from 'react-router-dom';
import { TaskHistoryTable } from '@/components/TaskHistoryTable';
import { useSamples } from '@/hooks/useSamples';
import { useTaskHistory } from '@/hooks/useTaskHistory';

export function TaskHistoryPage() {
  const navigate = useNavigate();
  const { data: samples } = useSamples();
  const { items, filters, setFilters } = useTaskHistory();

  return (
    <>
      <Typography.Title level={3}>{'\u4efb\u52a1\u5386\u53f2'}</Typography.Title>
      <TaskHistoryTable items={items} filters={filters} samples={samples} onFilterChange={setFilters} onOpen={(taskId) => navigate(`/tasks/${taskId}`)} />
    </>
  );
}
