import { Card, Col, Descriptions, Empty, Image, Row, Space, Statistic, Typography } from 'antd';
import type { TaskDetail, TaskResult } from '@/types/api';

interface Props {
  task?: TaskDetail;
  result?: TaskResult;
  loading?: boolean;
  mode: 'success' | 'failed' | 'cancelled';
}

export function ResultViewer({ task, result, loading, mode }: Props) {
  if (mode === 'failed') {
    return (
      <Card title={'\u4efb\u52a1\u5931\u8d25'} loading={loading}>
        <Typography.Title level={5}>{'\u9519\u8bef\u6458\u8981'}</Typography.Title>
        <Typography.Paragraph>{task?.error?.message ?? '\u4efb\u52a1\u5931\u8d25\uff0c\u8bf7\u67e5\u770b\u65e5\u5fd7\u4e86\u89e3\u8be6\u60c5\u3002'}</Typography.Paragraph>
        {task?.error?.details && <Typography.Paragraph type="secondary">{task.error.details}</Typography.Paragraph>}
      </Card>
    );
  }

  if (!result) {
    return (
      <Card title={'\u7ed3\u679c'} loading={loading}>
        <Empty description={'\u540e\u7aef\u672a\u8fd4\u56de\u7ed3\u679c\u6570\u636e'} />
      </Card>
    );
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title={mode === 'cancelled' ? '\u90e8\u5206\u7ed3\u679c' : '\u63a8\u7406\u7ed3\u679c'} loading={loading}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Typography.Title level={5}>{'\u8f93\u5165\u56fe\u7247\u5899'}</Typography.Title>
            <Image.PreviewGroup>
              <div className="image-wall">
                {(result.inputImages ?? task?.inputImages ?? []).map((url) => (
                  <Image key={url} src={url} alt="input" />
                ))}
              </div>
            </Image.PreviewGroup>
          </Col>
          <Col xs={24} lg={12}>
            <Typography.Title level={5}>{'\u89c6\u9891\u64ad\u653e\u5668'}</Typography.Title>
            {result.videoUrl ? <video src={result.videoUrl} controls className="result-video" /> : <Empty description={'\u6682\u65e0\u89c6\u9891\u8f93\u51fa'} />}
          </Col>
          <Col xs={24} lg={12}>
            <Typography.Title level={5}>{'\u6df1\u5ea6\u56fe'}</Typography.Title>
            <Image.PreviewGroup>
              <div className="image-wall">
                {(result.depthImages ?? []).map((url) => (
                  <Image key={url} src={url} alt="depth" />
                ))}
              </div>
            </Image.PreviewGroup>
          </Col>
          <Col xs={24} lg={12}>
            <Card size="small" title={'\u53c2\u6570'}>
              <Descriptions column={1} size="small">
                {Object.entries(result.parameters ?? task?.parameters ?? {}).map(([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    {String(value)}
                  </Descriptions.Item>
                ))}
              </Descriptions>
            </Card>
            <Card size="small" title={'\u6307\u6807'} style={{ marginTop: 16 }}>
              {!result.metrics?.length ? (
                <Empty description={'\u6682\u65e0\u6307\u6807\u6570\u636e'} />
              ) : (
                <Row gutter={[16, 16]}>
                  {result.metrics.map((metric) => (
                    <Col key={metric.label} xs={24} md={12}>
                      <Statistic title={metric.label} value={metric.value} suffix={metric.unit} />
                    </Col>
                  ))}
                </Row>
              )}
            </Card>
          </Col>
        </Row>
      </Card>
    </Space>
  );
}
