import { useMemo, useState } from 'react';
import { Alert, Button, Card, Col, Empty, Image, Row, Select, Space, Spin, Typography, message } from 'antd';
import type { UploadFile } from 'antd';
import { useCreateTask, useTaskDetail, useTaskResult } from '@/hooks/useTask';
import { usePresets, useSamples } from '@/hooks/useSamples';
import { ImageUploader } from '@/components/ImageUploader';
import { useSettings } from '@/hooks/useSettings';
import { isTaskTerminal } from '@/utils/task';

export function TaskCreatePage() {
  const { settings } = useSettings();
  const { data: samples, isLoading: samplesLoading } = useSamples();
  const { data: presets, isLoading: presetsLoading } = usePresets();
  const createTask = useCreateTask();
  const [sampleId, setSampleId] = useState<string>();
  const [presetId, setPresetId] = useState<string>();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [taskId, setTaskId] = useState<string>();

  const currentTaskQuery = useTaskDetail(taskId);
  const currentTask = currentTaskQuery.data;
  const resultQuery = useTaskResult(taskId, isTaskTerminal(currentTask?.state));

  const selectedPreset = presets?.find((item) => item.id === presetId);
  const selectedSample = samples?.find((item) => item.id === sampleId);

  const uploadedPreviewUrls = useMemo(
    () => fileList.map((file) => file.thumbUrl).filter(Boolean) as string[],
    [fileList],
  );

  const inputImageUrls = useMemo(() => {
    if (uploadedPreviewUrls.length > 0) return uploadedPreviewUrls;
    return selectedSample?.inputImages ?? selectedSample?.previewImages ?? [];
  }, [selectedSample, uploadedPreviewUrls]);

  const handleSubmit = async () => {
    const files = fileList.map((file) => file.originFileObj).filter(Boolean) as File[];

    if (!presetId) {
      message.error('\u8bf7\u5148\u9009\u62e9\u9884\u8bbe\u63a8\u7406\u811a\u672c\u3002');
      return;
    }

    if (!sampleId && files.length === 0) {
      message.error('\u8bf7\u5148\u9009\u62e9\u6837\u4f8b\u6216\u4e0a\u4f20\u8f93\u5165\u56fe\u50cf\u3002');
      return;
    }

    try {
      const response = await createTask.mutateAsync({
        sampleId,
        presetId,
        images: files,
        options: settings.taskOptions,
      });
      setTaskId(response.id);
      message.success(`\u4efb\u52a1 ${response.id} \u5df2\u521b\u5efa\u3002`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : '\u4efb\u52a1\u521b\u5efa\u5931\u8d25';
      message.error(msg);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>{'\u65b0\u5efa\u4efb\u52a1'}</Typography.Title>

      <Card title={'\u4efb\u52a1\u914d\u7f6e'}>
        <Row gutter={[16, 16]}>
          <Col xs={24} xl={12}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Typography.Title level={5} style={{ margin: 0 }}>{'\u9884\u8bbe\u63a8\u7406\u811a\u672c'}</Typography.Title>
              <Select
                loading={presetsLoading}
                placeholder={'\u8bf7\u9009\u62e9\u9884\u8bbe\u63a8\u7406\u811a\u672c'}
                value={presetId}
                onChange={setPresetId}
                optionFilterProp="label"
                showSearch
                style={{ width: '100%' }}
                options={presets?.map((preset) => ({
                  value: preset.id,
                  label: preset.name,
                }))}
              />
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Typography.Text strong>{'\u5f53\u524d\u914d\u7f6e'}</Typography.Text>
                <Typography.Text>{`Checkpoint\uff1a${selectedPreset?.checkpoint ?? '?'}`}</Typography.Text>
                <Typography.Text>{`Context Views\uff1a${selectedPreset?.contextViews ?? '?'}`}</Typography.Text>
                <Typography.Text>{`Image Shape\uff1a${selectedPreset?.imageShape?.length ? selectedPreset.imageShape.join(' ? ') : '?'}`}</Typography.Text>
              </Space>
            </Space>
          </Col>
          <Col xs={24} xl={12}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Typography.Title level={5} style={{ margin: 0 }}>{'\u4e0a\u4f20\u8f93\u5165\u56fe\u50cf'}</Typography.Title>
              <ImageUploader fileList={fileList} onChange={setFileList} />
            </Space>
          </Col>
        </Row>
      </Card>

      <Card
        title={'\u6837\u4f8b\u9009\u62e9'}
        extra={
          <Space>
            <Button type="primary" onClick={handleSubmit} loading={createTask.isPending}>
              {createTask.isPending ? '\u751f\u6210\u4e2d\u2026' : '\u63d0\u4ea4\u4efb\u52a1'}
            </Button>
            <Button onClick={() => taskId && currentTaskQuery.refetch()} disabled={!taskId || currentTaskQuery.isFetching}>
              {'\u5237\u65b0\u4efb\u52a1'}
            </Button>
          </Space>
        }
      >
        <Select
          showSearch
          allowClear
          placeholder={'\u8bf7\u9009\u62e9\u6837\u4f8b'}
          value={sampleId}
          onChange={setSampleId}
          optionFilterProp="label"
          style={{ width: '100%' }}
          loading={samplesLoading}
          options={samples?.map((sample) => ({
            value: sample.id,
            label: sample.name,
          }))}
        />
        {sampleId && (
          <Typography.Text type="secondary" style={{ display: 'block', marginTop: 12 }}>
            {selectedSample?.description ?? '\u5df2\u9009\u62e9\u6837\u4f8b'}
          </Typography.Text>
        )}
      </Card>

      <Row gutter={[16, 16]} align="stretch">
        <Col xs={24} xl={12}>
          <Card title={'\u8f93\u5165\u56fe\u50cf'} className="full-height-card">
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {!inputImageUrls.length ? (
                <Empty description={'\u6682\u65e0\u8f93\u5165\u56fe\u50cf\uff0c\u8bf7\u9009\u62e9\u6837\u4f8b\u6216\u4e0a\u4f20\u56fe\u7247\u3002'} />
              ) : (
                <Image.PreviewGroup>
                  <div className="preview-grid preview-grid-large">
                    {inputImageUrls.map((url) => (
                      <Image key={url} src={url} alt="input" className="preview-image" />
                    ))}
                  </div>
                </Image.PreviewGroup>
              )}
              {taskId && !isTaskTerminal(currentTask?.state) && <Alert type="info" showIcon message={`\u4efb\u52a1 ${taskId} \u751f\u6210\u4e2d`} />}
            </Space>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title={'\u751f\u6210\u7ed3\u679c'} className="full-height-card">
            {resultQuery.data?.videoUrl ? (
              <video src={resultQuery.data.videoUrl} controls className="result-video large-video" />
            ) : currentTask && !isTaskTerminal(currentTask.state) ? (
              <div className="video-placeholder"><Spin tip={'\u6b63\u5728\u751f\u6210\u89c6\u9891\u7ed3\u679c\u2026'} /></div>
            ) : (
              <Empty description={'\u6682\u65e0\u751f\u6210\u7ed3\u679c'} />
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
