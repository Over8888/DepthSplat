import type { ReactNode } from 'react';
import { Card, Col, Form, InputNumber, Row, Select, Switch } from 'antd';
import type { DepthSplatParameters } from '@/types/api';

interface Props {
  initialValues: DepthSplatParameters;
  onFinish: (values: DepthSplatParameters) => void;
  submitting?: boolean;
  children?: ReactNode;
}

export function ParameterForm({ initialValues, onFinish, submitting, children }: Props) {
  return (
    <Card title={'\u53c2\u6570\u8bbe\u7f6e'}>
      <Form layout="vertical" initialValues={initialValues} onFinish={onFinish} disabled={submitting}>
        <Row gutter={16}>
          <Col xs={24} md={8}>
            <Form.Item label={'\u63a8\u7406\u6b65\u6570'} name="numInferenceSteps" rules={[{ required: true, message: '\u5fc5\u586b\u9879' }]}>
              <InputNumber min={1} max={200} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col xs={24} md={8}>
            <Form.Item label={'\u5f15\u5bfc\u5f3a\u5ea6'} name="guidanceScale" rules={[{ required: true, message: '\u5fc5\u586b\u9879' }]}>
              <InputNumber min={0} max={50} step={0.5} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col xs={24} md={8}>
            <Form.Item label={'\u968f\u673a\u79cd\u5b50'} name="seed">
              <InputNumber min={0} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col xs={24} md={8}>
            <Form.Item label={'\u8f93\u51fa\u5e27\u7387'} name="outputFps" rules={[{ required: true, message: '\u5fc5\u586b\u9879' }]}>
              <InputNumber min={1} max={120} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col xs={24} md={8}>
            <Form.Item label={'\u8f93\u51fa\u683c\u5f0f'} name="outputFormat" rules={[{ required: true, message: '\u5fc5\u586b\u9879' }]}>
              <Select options={[{ value: 'mp4', label: 'MP4' }, { value: 'webm', label: 'WebM' }]} />
            </Form.Item>
          </Col>
          <Col xs={24} md={8}>
            <Form.Item label={'\u5bfc\u51fa\u6df1\u5ea6\u56fe'} name="exportDepthMap" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Col>
        </Row>
        {children}
      </Form>
    </Card>
  );
}
