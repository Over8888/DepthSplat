import { Badge, Card, Descriptions, Typography } from 'antd';
import type { TaskDetail } from '@/types/api';
import { TASK_STATE_LABELS } from '@/utils/constants';
import { formatDuration, formatTimestamp } from '@/utils/task';

interface Props {
  task?: TaskDetail;
  loading?: boolean;
  localState?: string;
}

const badgeStatusMap: Record<string, 'default' | 'processing' | 'success' | 'error' | 'warning'> = {
  queued: 'default',
  preparing: 'processing',
  running: 'processing',
  postprocessing: 'processing',
  success: 'success',
  failed: 'error',
  cancelled: 'warning',
  submitting: 'processing',
  loading: 'processing',
  cancelling: 'warning',
};

export function TaskStatusCard({ task, loading, localState }: Props) {
  const state = localState ?? task?.state ?? 'loading';
  const label = task?.state ? TASK_STATE_LABELS[task.state] : state;

  return (
    <Card title={'\u4efb\u52a1\u72b6\u6001'} loading={loading}>
      <Descriptions column={1} size="small">
        <Descriptions.Item label={'\u72b6\u6001'}>
          <Badge status={badgeStatusMap[state] ?? 'default'} text={label} />
        </Descriptions.Item>
        <Descriptions.Item label={'\u9636\u6bb5'}>{task?.stage ?? '?'}</Descriptions.Item>
        <Descriptions.Item label={'\u521b\u5efa\u65f6\u95f4'}>{formatTimestamp(task?.createdAt)}</Descriptions.Item>
        <Descriptions.Item label={'\u66f4\u65b0\u65f6\u95f4'}>{formatTimestamp(task?.updatedAt)}</Descriptions.Item>
        <Descriptions.Item label={'\u8017\u65f6'}>{formatDuration(task?.timings?.durationSeconds)}</Descriptions.Item>
      </Descriptions>
      {localState === 'cancelling' && <Typography.Text type="warning">{'\u5df2\u53d1\u9001\u53d6\u6d88\u8bf7\u6c42\uff0c\u7b49\u5f85\u540e\u7aef\u786e\u8ba4\u3002'}</Typography.Text>}
    </Card>
  );
}
