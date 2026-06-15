import { Badge, Card, Descriptions, Typography } from 'antd';
import type { TaskDetail } from '@/types/api';
import { TASK_STATE_LABELS } from '@/utils/constants';
import { formatDuration, formatTimestamp } from '@/utils/task';

interface Props {
  task?: TaskDetail;
  loading?: boolean;
  localState?: string;
}

const TEXT = {
  title: '\u4efb\u52a1\u72b6\u6001',
  state: '\u72b6\u6001',
  stage: '\u9636\u6bb5',
  createdAt: '\u521b\u5efa\u65f6\u95f4',
  updatedAt: '\u66f4\u65b0\u65f6\u95f4',
  duration: '\u8017\u65f6',
  dataLoad: '\u6570\u636e\u52a0\u8f7d',
  dataPrep: '\u6570\u636e\u51c6\u5907',
  inference: '\u63a8\u7406',
  splatConversion: 'Splat \u8f6c\u6362',
  cancelling: '\u5df2\u53d1\u9001\u53d6\u6d88\u8bf7\u6c42\uff0c\u7b49\u5f85\u540e\u7aef\u786e\u8ba4\u3002',
} as const;

const badgeStatusMap: Record<string, 'default' | 'processing' | 'success' | 'error' | 'warning'> = {
  queued: 'default',
  preparing: 'processing',
  running: 'processing',
  postprocessing: 'processing',
  success: 'success',
  failed: 'error',
  cancelled: 'warning',
  missing: 'default',
  submitting: 'processing',
  loading: 'processing',
  cancelling: 'warning',
};

export function TaskStatusCard({ task, loading, localState }: Props) {
  const state = localState ?? task?.state ?? 'loading';
  const label = task?.state ? TASK_STATE_LABELS[task.state] : state;
  const stageDurations = [
    { label: TEXT.dataLoad, value: task?.timings?.dataLoadSeconds },
    { label: TEXT.dataPrep, value: task?.timings?.dataPrepSeconds },
    { label: TEXT.inference, value: task?.timings?.inferenceSeconds },
    { label: TEXT.splatConversion, value: task?.timings?.splatConversionSeconds },
  ].filter((item) => item.value != null);

  return (
    <Card title={TEXT.title} loading={loading} className="full-height-card">
      <div style={{ minHeight: 320, display: 'flex', flexDirection: 'column' }}>
        <Descriptions column={1} size="small">
          <Descriptions.Item label={TEXT.state}>
            <Badge status={badgeStatusMap[state] ?? 'default'} text={label} />
          </Descriptions.Item>
          <Descriptions.Item label={TEXT.stage}>{task?.stage ?? '?'}</Descriptions.Item>
          <Descriptions.Item label={TEXT.createdAt}>{formatTimestamp(task?.createdAt)}</Descriptions.Item>
          <Descriptions.Item label={TEXT.updatedAt}>{formatTimestamp(task?.updatedAt)}</Descriptions.Item>
        </Descriptions>

        {(task?.timings?.durationSeconds != null || stageDurations.length > 0) && (
          <Descriptions column={1} size="small" style={{ marginTop: 'auto', paddingTop: 20 }}>
            <Descriptions.Item label={TEXT.duration}>{formatDuration(task?.timings?.durationSeconds)}</Descriptions.Item>
            {stageDurations.map((item) => (
              <Descriptions.Item key={item.label} label={item.label}>
                {formatDuration(item.value)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        )}

        {localState === 'cancelling' && <Typography.Text type="warning">{TEXT.cancelling}</Typography.Text>}
      </div>
    </Card>
  );
}
