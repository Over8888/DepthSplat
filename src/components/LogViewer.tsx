import { Card, Empty, List, Tag, Typography } from 'antd';
import type { TaskLogEntry } from '@/types/api';
import { formatTimestamp } from '@/utils/task';

interface Props {
  entries?: TaskLogEntry[];
  loading?: boolean;
}

const levelColorMap = {
  debug: 'default',
  info: 'blue',
  warning: 'orange',
  error: 'red',
} as const;

export function LogViewer({ entries, loading }: Props) {
  return (
    <Card title={'\u5b9e\u65f6\u65e5\u5fd7'} loading={loading} styles={{ body: { maxHeight: 420, overflow: 'auto' } }}>
      {!entries?.length ? (
        <Empty description={'\u540e\u7aef\u6682\u672a\u8fd4\u56de\u65e5\u5fd7'} />
      ) : (
        <List
          dataSource={entries}
          renderItem={(entry) => (
            <List.Item>
              <div className="log-row">
                <Typography.Text type="secondary">{formatTimestamp(entry.timestamp)}</Typography.Text>
                {entry.level && <Tag color={levelColorMap[entry.level]}>{entry.level.toUpperCase()}</Tag>}
                <Typography.Text code>{entry.message}</Typography.Text>
              </div>
            </List.Item>
          )}
        />
      )}
    </Card>
  );
}
