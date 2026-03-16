import { Button, Card, Form, Input, InputNumber, Select, Switch } from 'antd';
import type { SettingsState } from '@/types/ui';

interface Props {
  initialValues: SettingsState;
  onSubmit: (values: SettingsState) => void;
}

export function SettingsForm({ initialValues, onSubmit }: Props) {
  return (
    <Card title={'\u524d\u7aef\u8bbe\u7f6e'}>
      <Form layout="vertical" initialValues={initialValues} onFinish={onSubmit}>
        <Form.Item label={'\u540e\u7aef\u57fa\u7840\u5730\u5740'} name="backendBaseUrl" rules={[{ required: true, message: '\u540e\u7aef\u5730\u5740\u4e0d\u80fd\u4e3a\u7a7a' }]}>
          <Input placeholder="http://127.0.0.1:8000" />
        </Form.Item>
        <Form.Item label={'\u9ed8\u8ba4\u63a8\u7406\u6b65\u6570'} name={['defaultParameters', 'numInferenceSteps']}>
          <InputNumber min={1} max={200} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item label={'\u9ed8\u8ba4\u5f15\u5bfc\u5f3a\u5ea6'} name={['defaultParameters', 'guidanceScale']}>
          <InputNumber min={0} max={50} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item label={'\u9ed8\u8ba4\u5e27\u7387'} name={['defaultParameters', 'outputFps']}>
          <InputNumber min={1} max={120} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item label={'\u9ed8\u8ba4\u8f93\u51fa\u683c\u5f0f'} name={['defaultParameters', 'outputFormat']}>
          <Select options={[{ value: 'mp4', label: 'MP4' }, { value: 'webm', label: 'WebM' }]} />
        </Form.Item>
        <Form.Item label={'\u9ed8\u8ba4\u5bfc\u51fa\u6df1\u5ea6\u56fe'} name={['defaultParameters', 'exportDepthMap']} valuePropName="checked">
          <Switch />
        </Form.Item>
        <Form.Item label={'\u65e5\u5fd7\u81ea\u52a8\u6eda\u52a8'} name={['preferences', 'autoScrollLogs']} valuePropName="checked">
          <Switch />
        </Form.Item>
        <Form.Item label={'\u7d27\u51d1\u5386\u53f2\u5217\u8868'} name={['preferences', 'compactHistory']} valuePropName="checked">
          <Switch />
        </Form.Item>
        <Button type="primary" htmlType="submit">
          {'\u4fdd\u5b58\u8bbe\u7f6e'}
        </Button>
      </Form>
    </Card>
  );
}
