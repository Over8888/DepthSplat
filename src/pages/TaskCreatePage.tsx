import { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Card, Col, Descriptions, Image, InputNumber, Row, Select, Space, Typography, message } from 'antd';
import type { UploadFile } from 'antd';
import { VideoUploader } from '@/components/VideoUploader';
import { useCreateTask, useTaskDetail } from '@/hooks/useTask';
import { usePresets, useSamples } from '@/hooks/useSamples';
import { setCurrentTaskState, useCurrentTask } from '@/store/currentTaskStore';
import { useSettings } from '@/hooks/useSettings';

const TEXT = {
  createTitle: '\u65b0\u5efa\u4efb\u52a1',
  configTitle: '\u4efb\u52a1\u914d\u7f6e',
  presetTitle: '\u6a21\u5f0f\u9009\u62e9',
  inputViewSelector: '\u8f93\u5165\u89c6\u89d2',
  presetPlaceholder: '\u8bf7\u9009\u62e9\u6a21\u5f0f',
  currentConfig: '\u5f53\u524d\u914d\u7f6e',
  sampleTitle: '\u6837\u4f8b\u9009\u62e9',
  submitLoading: '\u751f\u6210\u4e2d\u2026',
  submit: '\u63d0\u4ea4\u4efb\u52a1',
  refresh: '\u5237\u65b0\u4efb\u52a1',
  samplePlaceholder: '\u8bf7\u9009\u62e9\u6837\u4f8b',
  samplePlaceholderDisabled: '\u8bf7\u5148\u9009\u62e9\u6a21\u5f0f',
  samplePlaceholderVideoLocked: '\u5df2\u624b\u52a8\u8f93\u5165\u89c6\u9891',
  selectPresetFirst: '\u8bf7\u5148\u9009\u62e9\u6a21\u5f0f\u3002',
  selectInputSource: '\u8bf7\u9009\u62e9\u6837\u4f8b\u6216\u4e0a\u4f20\u4e00\u4e2a\u89c6\u9891\u3002',
  createFailed: '\u4efb\u52a1\u521b\u5efa\u5931\u8d25',
  checkpoint: 'mode',
  contextViews: 'Context Views',
  inputViews: '\u8f93\u5165\u89c6\u89d2\u6570\u91cf',
  targetViews: '\u76ee\u6807\u89c6\u89d2\u6570\u91cf',
  sceneNumber: '\u573a\u666f\u7f16\u53f7',
  scenePreview: '\u573a\u666f\u9884\u89c8',
  noScenePreview: '\u6682\u65e0\u573a\u666f\u9884\u89c8\u56fe',
  imageShape: 'Image Shape',
  taskCreated: (id: string) => `\u4efb\u52a1 ${id} \u5df2\u521b\u5efa\u3002`,
} as const;

const MODE_OPTIONS = [
  { key: 'large', label: '\u9ad8\u8d28\u91cf\u6a21\u5f0f' },
  { key: 'base', label: '\u5747\u8861\u6a21\u5f0f' },
  { key: 'small', label: '\u5feb\u901f\u6a21\u5f0f' },
] as const;

type ModeKey = (typeof MODE_OPTIONS)[number]['key'];

const INPUT_VIEW_OPTIONS = [2, 4, 6] as const;
type InputViewCount = (typeof INPUT_VIEW_OPTIONS)[number];

const getNearestInputViewCount = (value: number | null): InputViewCount => {
  if (!value) return 2;
  return INPUT_VIEW_OPTIONS.reduce((nearest, option) =>
    Math.abs(option - value) < Math.abs(nearest - value) ? option : nearest,
  );
};

const getModeLabel = (checkpoint?: string): string => {
  if (!checkpoint) return '?';
  const lower = checkpoint.toLowerCase();
  return MODE_OPTIONS.find((mode) => lower.includes(mode.key))?.label ?? lower;
};

const getPresetSearchText = (preset: { id?: string; name?: string; checkpoint?: string }) =>
  [preset.id, preset.name, preset.checkpoint].filter(Boolean).join(' ').toLowerCase();

const hasModeToken = (text: string, mode: ModeKey) => new RegExp(`(?:^|[_\\-\\s])${mode}(?:$|[_\\-\\s])`).test(text);

const inferPresetMode = (preset: { id?: string; name?: string; checkpoint?: string }): ModeKey | undefined => {
  const text = getPresetSearchText(preset);
  const exactMode = MODE_OPTIONS.find((mode) => hasModeToken(text, mode.key));
  if (exactMode) return exactMode.key;
  if (/均衡|balance|balanced/.test(text)) return 'base';
  if (/快速|fast/.test(text)) return 'small';
  if (/高质量/.test(text)) return 'large';
  return undefined;
};

const inferPresetInputViewCount = (preset: { id?: string; name?: string; checkpoint?: string; contextViews?: number }): number | undefined => {
  const text = [preset.id, preset.name, preset.checkpoint].filter(Boolean).join(' ').toLowerCase();
  const match = text.match(/(?:^|[_\-\s])(2|4|6)\s*view(?:$|[_\-\s])/);
  if (match) return Number(match[1]);

  if (preset.contextViews && INPUT_VIEW_OPTIONS.includes(preset.contextViews as InputViewCount)) return preset.contextViews;
  return undefined;
};

const buildPresetId = (mode: ModeKey, viewCount: InputViewCount) => `re10k_${mode}_${viewCount}view`;

const isModeMatch = (preset: { id?: string; name?: string; checkpoint?: string }, mode: ModeKey) =>
  inferPresetMode(preset) === mode;

const getSampleDisplayName = (sample: { sceneNumber?: string | number; name?: string; id?: string }) => {
  const rawName = String(sample.sceneNumber ?? sample.name ?? sample.id ?? '');
  return rawName.replace(/\s*\([^)]*\)\s*$/, '');
};

export function TaskCreatePage() {
  const { settings } = useSettings();
  const currentTaskSession = useCurrentTask();
  const [presetId, setPresetId] = useState<string | undefined>(currentTaskSession.presetId);
  const [sampleId, setSampleId] = useState<string | undefined>(currentTaskSession.sampleId);
  const [selectedInputViewCount, setSelectedInputViewCount] = useState<InputViewCount>(2);
  const [uploadedVideoFiles, setUploadedVideoFiles] = useState<UploadFile[]>([]);
  const previousPresetIdRef = useRef<string | undefined>(currentTaskSession.presetId);

  const { data: samples, isLoading: samplesLoading } = useSamples(presetId);
  const { data: presets, isLoading: presetsLoading } = usePresets();
  const createTask = useCreateTask();

  const taskId = currentTaskSession.taskId;
  const currentTaskQuery = useTaskDetail(taskId);

  const modePresetOptions = MODE_OPTIONS.map((mode) => {
    const preset = presets?.find((item) => isModeMatch(item, mode.key) && inferPresetInputViewCount(item) === selectedInputViewCount);
    return {
      id: preset?.id ?? buildPresetId(mode.key, selectedInputViewCount),
      mode,
      preset,
    };
  });
  const selectedModePreset = modePresetOptions.find((item) => item.id === presetId);
  const selectedPreset = selectedModePreset?.preset ?? presets?.find((item) => item.id === presetId);
  const selectedPresetInputViewCount = selectedPreset ? inferPresetInputViewCount(selectedPreset) : undefined;
  const selectedPresetMode = selectedPreset ? inferPresetMode(selectedPreset) : undefined;
  const selectedModeLabel = selectedModePreset?.mode.label ?? MODE_OPTIONS.find((mode) => mode.key === selectedPresetMode)?.label;
  const selectedSample = samples?.find((item) => item.id === sampleId);
  const samplePreviewImage = selectedSample?.thumbnailUrl ?? selectedSample?.previewImages?.[0] ?? selectedSample?.inputImages?.[0];
  const inputViewCount = selectedPresetInputViewCount ?? selectedInputViewCount;
  const targetViewCount = selectedSample?.targetViewCount ?? selectedPreset?.targetViews;

  useEffect(() => {
    setPresetId(currentTaskSession.presetId);
  }, [currentTaskSession.presetId]);
  useEffect(() => {
    setSampleId(currentTaskSession.sampleId);
  }, [currentTaskSession.sampleId]);

  useEffect(() => {
    if (previousPresetIdRef.current === presetId) return;
    previousPresetIdRef.current = presetId;
    setSampleId(undefined);
    setCurrentTaskState({ presetId, sampleId: undefined });
  }, [presetId]);

  useEffect(() => {
    if (!selectedPresetInputViewCount) return;
    if (!INPUT_VIEW_OPTIONS.includes(selectedPresetInputViewCount as InputViewCount)) return;
    if (selectedPresetInputViewCount !== selectedInputViewCount) {
      setSelectedInputViewCount(selectedPresetInputViewCount as InputViewCount);
    }
  }, [selectedPresetInputViewCount, selectedInputViewCount]);

  const uploadedVideoFile = useMemo<File | undefined>(
    () => uploadedVideoFiles.find((file) => file.originFileObj)?.originFileObj,
    [uploadedVideoFiles],
  );

  const handleSubmit = async () => {
    const hasVideoInput = !!uploadedVideoFile;

    if (!presetId && !hasVideoInput) {
      message.error(TEXT.selectPresetFirst);
      return;
    }

    if (!sampleId && !hasVideoInput) {
      message.error(TEXT.selectInputSource);
      return;
    }

    try {
      const response = await createTask.mutateAsync({
        sampleId: hasVideoInput ? undefined : sampleId,
        preset: presetId,
        inputViewCount,
        video: uploadedVideoFile,
        options: settings.taskOptions,
      });
      setCurrentTaskState({
        taskId: response.id,
        presetId,
        sampleId: hasVideoInput ? undefined : sampleId,
      });
      message.success(TEXT.taskCreated(response.id));
    } catch (error) {
      const msg = error instanceof Error ? error.message : TEXT.createFailed;
      message.error(msg);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>{TEXT.createTitle}</Typography.Title>

      <Card title={TEXT.configTitle}>
        <Row gutter={[16, 16]}>
          <Col xs={24} xl={12}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Space direction="vertical" size={8} style={{ width: '100%' }}>
                <Space align="center" wrap>
                  <Typography.Title level={5} style={{ margin: 0 }}>
                    {TEXT.presetTitle}
                  </Typography.Title>
                  <Space size={6} align="center">
                    <Typography.Text type="secondary">{TEXT.inputViewSelector}</Typography.Text>
                    <InputNumber
                      className="input-view-stepper"
                      min={2}
                      max={6}
                      step={2}
                      value={selectedInputViewCount}
                      onChange={(value) => {
                        const nextValue = getNearestInputViewCount(value);
                        if (nextValue !== selectedInputViewCount) {
                          setPresetId(undefined);
                          setSampleId(undefined);
                          setCurrentTaskState({ presetId: undefined, sampleId: undefined });
                        }
                        setSelectedInputViewCount(nextValue);
                      }}
                      controls
                      keyboard={false}
                    />
                  </Space>
                </Space>

              </Space>
              <Select
                loading={presetsLoading}
                placeholder={TEXT.presetPlaceholder}
                value={presetId}
                onChange={(value) => {
                  setPresetId(value);
                }}
                optionFilterProp="label"
                showSearch
                style={{ width: '100%' }}
                options={modePresetOptions.map((item) => ({
                  value: item.id,
                  label: item.mode.label,
                }))}
              />
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Typography.Text strong>{TEXT.currentConfig}</Typography.Text>
                <Typography.Text>{`${TEXT.checkpoint}\uff1a${selectedModeLabel ?? '?'}`}</Typography.Text>
                <Typography.Text>{`${TEXT.contextViews}\uff1a${selectedPreset?.contextViews ?? '?'}`}</Typography.Text>
                <Typography.Text>{`${TEXT.inputViews}\uff1a${inputViewCount ?? '?'}`}</Typography.Text>
                <Typography.Text>{`${TEXT.targetViews}\uff1a${targetViewCount ?? '?'}`}</Typography.Text>
                <Typography.Text>{`${TEXT.imageShape}\uff1a${selectedPreset?.imageShape?.length ? selectedPreset.imageShape.join(' \u00d7 ') : '?'}`}</Typography.Text>
              </Space>
            </Space>
          </Col>
          <Col xs={24} xl={12}>
            <VideoUploader
              fileList={uploadedVideoFiles}
              onChange={(files) => {
                setUploadedVideoFiles(files);
                if (files.length) {
                  setSampleId(undefined);
                  setCurrentTaskState({ presetId, sampleId: undefined });
                }
              }}
            />
          </Col>
        </Row>
      </Card>

      <Card
        title={TEXT.sampleTitle}
        extra={
          <Space>
            <Button type="primary" onClick={handleSubmit} loading={createTask.isPending}>
              {createTask.isPending ? TEXT.submitLoading : TEXT.submit}
            </Button>
            <Button onClick={() => taskId && currentTaskQuery.refetch()} disabled={!taskId || currentTaskQuery.isFetching}>
              {TEXT.refresh}
            </Button>
          </Space>
        }
      >
        <Select
          showSearch
          allowClear
          placeholder={uploadedVideoFile ? TEXT.samplePlaceholderVideoLocked : presetId ? TEXT.samplePlaceholder : TEXT.samplePlaceholderDisabled}
          value={sampleId}
          onChange={(value) => {
            setSampleId(value);
            if (value) {
              setUploadedVideoFiles([]);
            }
            setCurrentTaskState({ presetId, sampleId: value });
          }}
          optionFilterProp="label"
          style={{ width: '100%' }}
          loading={samplesLoading}
          disabled={!presetId || Boolean(uploadedVideoFile)}
          options={samples?.map((sample) => ({
            value: sample.id,
            label: getSampleDisplayName(sample),
          }))}
        />
        {sampleId ? (
          <Space direction="vertical" size="middle" style={{ width: '100%', marginTop: 12 }}>
            <Card size="small" title={TEXT.scenePreview}>
              <Row gutter={[16, 16]} align="middle">
                <Col xs={24} md={10}>
                  {samplePreviewImage ? (
                    <Image src={samplePreviewImage} alt={selectedSample?.name ?? 'scene preview'} width="100%" height={180} style={{ objectFit: 'cover', borderRadius: 8 }} />
                  ) : (
                    <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f5f5f5', borderRadius: 8 }}>
                      <Typography.Text type="secondary">{TEXT.noScenePreview}</Typography.Text>
                    </div>
                  )}
                </Col>
                <Col xs={24} md={14}>
                  <Descriptions size="small" column={1}>
                    <Descriptions.Item label={TEXT.sceneNumber}>{selectedSample?.sceneNumber ?? selectedSample?.id ?? '?'}</Descriptions.Item>
                    <Descriptions.Item label={TEXT.inputViews}>{inputViewCount ?? '?'}</Descriptions.Item>
                    <Descriptions.Item label={TEXT.targetViews}>{targetViewCount ?? '?'}</Descriptions.Item>
                  </Descriptions>
                </Col>
              </Row>
            </Card>
          </Space>
        ) : null}
      </Card>
    </Space>
  );
}
