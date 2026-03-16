import { Card, Empty, Select, Space, Spin, Typography } from 'antd';
import type { SampleItem } from '@/types/api';

interface Props {
  samples?: SampleItem[];
  loading?: boolean;
  value?: string;
  onChange: (value?: string) => void;
}

export function SampleSelector({ samples, loading, value, onChange }: Props) {
  return (
    <Card title={'\u6837\u4f8b\u9009\u62e9'}>
      <Spin spinning={loading}>
        {!samples?.length ? (
          <Empty description={'\u540e\u7aef\u672a\u8fd4\u56de\u6837\u4f8b\u6570\u636e'} />
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Select
              showSearch
              allowClear
              placeholder={'\u8bf7\u9009\u62e9\u6837\u4f8b'}
              value={value}
              onChange={onChange}
              optionFilterProp="label"
              options={samples.map((sample) => ({
                value: sample.id,
                label: sample.name,
              }))}
            />
            {value && (
              <Typography.Text type="secondary">
                {samples.find((sample) => sample.id === value)?.description ?? '\u5df2\u9009\u62e9\u6837\u4f8b'}
              </Typography.Text>
            )}
          </Space>
        )}
      </Spin>
    </Card>
  );
}
