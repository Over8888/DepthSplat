import { Button, Card, DatePicker, Select, Space, Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import type { BackendTaskState, PresetScriptItem, TaskHistoryFilters, TaskHistoryItem } from '@/types/api';
import { TASK_STATE_LABELS } from '@/utils/constants';
import { formatTimestamp } from '@/utils/task';

interface Props {
  items: TaskHistoryItem[];
  filters: TaskHistoryFilters;
  presets?: PresetScriptItem[];
  onFilterChange: (filters: TaskHistoryFilters) => void;
  onOpen: (taskId: string) => void;
}

const TEXT = {
  uploadedVideo: '\u624b\u52a8\u4e0a\u4f20\u89c6\u9891',
} as const;

export function TaskHistoryTable({ items, filters, presets, onFilterChange, onOpen }: Props) {
  const columns: ColumnsType<TaskHistoryItem> = [
    { title: '\u4efb\u52a1 ID', dataIndex: 'id', key: 'id' },
    {
      title: '\u4efb\u52a1',
      dataIndex: 'sampleName',
      key: 'sampleName',
      render: (_, record) => record.sampleName || record.sampleId || TEXT.uploadedVideo,
    },
    {
      title: '\u72b6\u6001',
      dataIndex: 'state',
      key: 'state',
      render: (state: BackendTaskState) => <Tag>{TASK_STATE_LABELS[state]}</Tag>,
    },
    { title: '\u521b\u5efa\u65f6\u95f4', dataIndex: 'createdAt', key: 'createdAt', render: formatTimestamp },
    {
      title: '\u64cd\u4f5c',
      key: 'action',
      render: (_, record) => (
        <Button type="link" onClick={() => onOpen(record.id)}>
          {'\u67e5\u770b'}
        </Button>
      ),
    },
  ];

  return (
    <Card title={'\u4efb\u52a1\u5386\u53f2'} extra={<Tag>{'\u4ec5\u672c\u5730\u5386\u53f2'}</Tag>}>
      <Space wrap style={{ marginBottom: 16 }}>
        <Select
          value={filters.state ?? 'all'}
          style={{ width: 180 }}
          onChange={(value) => onFilterChange({ ...filters, state: value })}
          options={[
            { value: 'all', label: '\u5168\u90e8\u72b6\u6001' },
            { value: 'queued', label: '\u6392\u961f\u4e2d' },
            { value: 'preparing', label: '\u51c6\u5907\u4e2d' },
            { value: 'running', label: '\u8fd0\u884c\u4e2d' },
            { value: 'postprocessing', label: '\u540e\u5904\u7406\u4e2d' },
            { value: 'success', label: '\u6210\u529f' },
            { value: 'failed', label: '\u5931\u8d25' },
            { value: 'cancelled', label: '\u5df2\u53d6\u6d88' },
            { value: 'missing', label: '\u5df2\u4e22\u5931' },
          ]}
        />
        <Select
          allowClear
          placeholder={'\u6309\u63a8\u7406\u811a\u672c\u7b5b\u9009'}
          value={filters.presetId}
          style={{ width: 220 }}
          onChange={(value) => onFilterChange({ ...filters, presetId: value })}
          options={presets?.map((preset) => ({ value: preset.id, label: preset.name }))}
        />
        <DatePicker.RangePicker
          value={filters.timeRange ? [dayjs(filters.timeRange[0]), dayjs(filters.timeRange[1])] : null}
          onChange={(dates) =>
            onFilterChange({
              ...filters,
              timeRange: dates ? [dates[0]!.startOf('day').toISOString(), dates[1]!.endOf('day').toISOString()] : undefined,
            })
          }
        />
      </Space>
      <Table rowKey="id" columns={columns} dataSource={items} pagination={{ pageSize: 10 }} />
    </Card>
  );
}
