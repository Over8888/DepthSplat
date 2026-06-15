import { Card, Empty, Space, Tag, Typography } from 'antd';
import type { TaskLogEntry, TaskStageTiming } from '@/types/api';
import { formatDuration, formatTimestamp } from '@/utils/task';

interface Props {
  entries?: TaskLogEntry[];
  timings?: TaskStageTiming;
  loading?: boolean;
  refreshing?: boolean;
}

const levelColorMap = {
  debug: 'default',
  info: 'blue',
  warning: 'orange',
  error: 'red',
} as const;

const TEXT = {
  title: '\u5b9e\u65f6\u65e5\u5fd7',
  refreshing: '\u66f4\u65b0\u4e2d\u2026',
  empty: '\u540e\u7aef\u6682\u672a\u8fd4\u56de\u65e5\u5fd7',
  timingTitle: '\u9636\u6bb5\u8017\u65f6',
  dataLoad: '\u6570\u636e\u52a0\u8f7d',
  dataPrep: '\u6570\u636e\u51c6\u5907',
  inference: '\u63a8\u7406',
  splatConversion: 'Splat \u8f6c\u6362',
  colon: '\uff1a',
} as const;

const isProgressMessage = (message: string) =>
  /\d{1,3}%/.test(message) ||
  /it\/s/i.test(message) ||
  /\|\s*\d+\/\d+\s*\[/.test(message) ||
  /dataloader/i.test(message);

const isSummaryMessage = (message: string) =>
  /saving outputs to/i.test(message) ||
  /loaded pretrained weights/i.test(message);

const buildProgressKey = (message: string) =>
  message
    .replace(/\d{1,3}%/g, '')
    .replace(/[\u2588\u258f\u258e\u258d\u258c\u258b\u258a\u2586]+/g, '')
    .replace(/\[[^\]]*\]/g, '')
    .replace(/\b\d+\/\d+\b/g, '')
    .replace(/\b\d+(\.\d+)?it\/s\b/gi, '')
    .replace(/\|+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

const stripAnsi = (message: string) => message.replace(/\x1b\[[0-9;]*m/g, '');

export function LogViewer({ entries, timings, loading, refreshing }: Props) {
  const progressEntries = Array.from(
    (entries ?? [])
      .map((entry) => ({ ...entry, message: stripAnsi(entry.message) }))
      .filter((entry) => entry.message && isProgressMessage(entry.message))
      .reduce((acc, entry) => {
        acc.set(buildProgressKey(entry.message), entry);
        return acc;
      }, new Map<string, TaskLogEntry>())
      .values(),
  ).slice(-1);

  const summaryEntries = (entries ?? [])
    .map((entry) => ({ ...entry, message: stripAnsi(entry.message) }))
    .filter((entry) => entry.message && isSummaryMessage(entry.message))
    .slice(-2);

  const timingItems = [
    { label: TEXT.dataLoad, value: timings?.dataLoadSeconds },
    { label: TEXT.dataPrep, value: timings?.dataPrepSeconds },
    { label: TEXT.inference, value: timings?.inferenceSeconds },
    { label: TEXT.splatConversion, value: timings?.splatConversionSeconds },
  ].filter((item) => item.value != null);

  const hasContent = progressEntries.length > 0 || summaryEntries.length > 0 || timingItems.length > 0;

  return (
    <Card
      title={TEXT.title}
      loading={loading}
      className="full-height-card"
      extra={refreshing ? <Typography.Text type="secondary">{TEXT.refreshing}</Typography.Text> : null}
      styles={{ body: { maxHeight: 420, overflow: 'auto' } }}
    >
      {!hasContent ? (
        <Empty description={TEXT.empty} />
      ) : (
        <div>
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {progressEntries.map((entry) => (
              <div key={entry.id} className="log-row">
                <Typography.Text type="secondary">{formatTimestamp(entry.timestamp)}</Typography.Text>
                {entry.level && <Tag color={levelColorMap[entry.level]}>{entry.level.toUpperCase()}</Tag>}
                <Typography.Text code>{entry.message}</Typography.Text>
              </div>
            ))}

            {summaryEntries.map((entry) => (
              <div key={entry.id} className="log-row">
                <Typography.Text type="secondary">{formatTimestamp(entry.timestamp)}</Typography.Text>
                {entry.level && <Tag color={levelColorMap[entry.level]}>{entry.level.toUpperCase()}</Tag>}
                <Typography.Text>{entry.message}</Typography.Text>
              </div>
            ))}
          </Space>

          {timingItems.length > 0 && (
            <Space direction="vertical" size={4} style={{ width: '100%', marginTop: 50 }}>
              <Typography.Text strong>{TEXT.timingTitle}</Typography.Text>
              {timingItems.map((item) => (
                <Typography.Text key={item.label} type="secondary">
                  {`${item.label}${TEXT.colon}${formatDuration(item.value)}`}
                </Typography.Text>
              ))}
            </Space>
          )}
        </div>
      )}
    </Card>
  );
}
