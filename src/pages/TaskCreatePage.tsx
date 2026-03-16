import { useState } from 'react';
import { Button, Col, Form, Row, Space, Typography, message } from 'antd';
import type { UploadFile } from 'antd';
import { useNavigate } from 'react-router-dom';
import { ImageUploader } from '@/components/ImageUploader';
import { ParameterForm } from '@/components/ParameterForm';
import { SampleSelector } from '@/components/SampleSelector';
import { useCreateTask } from '@/hooks/useTask';
import { useSamples } from '@/hooks/useSamples';
import { useSettings } from '@/hooks/useSettings';
import type { DepthSplatParameters } from '@/types/api';

export function TaskCreatePage() {
  const navigate = useNavigate();
  const { settings } = useSettings();
  const { data: samples, isLoading: samplesLoading } = useSamples();
  const createTask = useCreateTask();
  const [sampleId, setSampleId] = useState<string>();
  const [fileList, setFileList] = useState<UploadFile[]>([]);

  const handleSubmit = async (parameters: DepthSplatParameters) => {
    const files = fileList.map((file) => file.originFileObj).filter(Boolean) as File[];

    if (files.length === 0) {
      message.error('\u8bf7\u81f3\u5c11\u4e0a\u4f20\u4e00\u5f20\u56fe\u7247\u3002');
      return;
    }

    try {
      const response = await createTask.mutateAsync({ sampleId, images: files, parameters });
      message.success(`\u4efb\u52a1 ${response.id} \u5df2\u521b\u5efa\u3002`);
      navigate(`/tasks/${response.id}`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : '\u4efb\u52a1\u521b\u5efa\u5931\u8d25';
      message.error(msg);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>{'\u521b\u5efa\u4efb\u52a1'}</Typography.Title>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <SampleSelector samples={samples} loading={samplesLoading} value={sampleId} onChange={setSampleId} />
        </Col>
        <Col xs={24} lg={14}>
          <Form layout="vertical">
            <ImageUploader fileList={fileList} onChange={setFileList} />
          </Form>
        </Col>
      </Row>
      <ParameterForm initialValues={{ ...settings.defaultParameters, seed: undefined }} onFinish={handleSubmit} submitting={createTask.isPending}>
        <Button type="primary" htmlType="submit" loading={createTask.isPending}>
          {'\u63d0\u4ea4\u4efb\u52a1'}
        </Button>
      </ParameterForm>
    </Space>
  );
}
