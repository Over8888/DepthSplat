import { Card, Empty, Radio, Space, Spin, Typography } from 'antd';
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
          <Radio.Group value={value} onChange={(event) => onChange(event.target.value)} style={{ width: '100%' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {samples.map((sample) => (
                <Card.Grid key={sample.id} style={{ width: '100%', cursor: 'pointer' }} onClick={() => onChange(sample.id)}>
                  <Radio value={sample.id}>
                    <Space direction="vertical" size={0}>
                      <Typography.Text strong>{sample.name}</Typography.Text>
                      <Typography.Text type="secondary">{sample.description || sample.id}</Typography.Text>
                    </Space>
                  </Radio>
                </Card.Grid>
              ))}
            </Space>
          </Radio.Group>
        )}
      </Spin>
    </Card>
  );
}
