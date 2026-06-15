import { Button, Card, Col, Empty, Image, Row, Segmented, Space, Statistic, Typography } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTaskDetail, useTaskInputImages, useTaskResult } from '@/hooks/useTask';
import { isTaskTerminal } from '@/utils/task';
import { InputImageGrid } from '@/components/InputImageGrid';
import { CameraFrustumViewer } from '@/components/CameraFrustumViewer';
import type { InputImageCameraInfo, JsonValue } from '@/types/api';

const VIEWER_BASE_URL = 'https://antimatter15.com/splat/';

const NO_TASK_MSG = '\u8bf7\u4ece\u5386\u53f2\u8bb0\u5f55\u4e2d\u67e5\u770b\u4efb\u52a1\u7ed3\u679c';

type ViewMode = 'predicted' | 'gt' | 'error';

const stripTrailingParenthetical = (value: string) => value.replace(/\s*\([^)]*\)\s*$/, '');

interface NovelViewCardProps {
  hasTask: boolean;
  loading: boolean;
  error: boolean | undefined;
  renderedImages?: string[];
  gtImages?: string[];
  errorImages?: string[];
}

const toNumber = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : undefined);

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const parseJsonValue = (value: JsonValue | undefined): unknown => {
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const normalizeCameraEntries = (value: JsonValue | undefined) => {
  const parsed = parseJsonValue(value);
  return Array.isArray(parsed) ? parsed : parsed === undefined ? [] : [parsed];
};

const parseRotation = (value: unknown): number[][] | undefined => {
  if (!Array.isArray(value)) return undefined;
  if (value.length === 9 && value.every((item) => typeof item === 'number')) {
    return [value.slice(0, 3), value.slice(3, 6), value.slice(6, 9)] as number[][];
  }
  if (value.length >= 3 && value.slice(0, 3).every((row) => Array.isArray(row))) {
    return value.slice(0, 3).map((row) => (row as unknown[]).slice(0, 3).map((item) => toNumber(item) ?? 0));
  }
  return undefined;
};

const parseTranslation = (value: unknown): number[] | undefined => {
  if (!Array.isArray(value)) return undefined;
  const items = value.slice(0, 3).map(toNumber);
  return items.every((item) => item !== undefined) ? (items as number[]) : undefined;
};

const parseExtrinsicEntry = (value: unknown) => {
  const record = asRecord(value);
  if (Object.keys(record).length) {
    const rotation = parseRotation(record.rotation ?? record.R ?? record.r);
    const translation = parseTranslation(record.translation ?? record.t ?? record.position);
    if (rotation && translation) return { rotation, translation };
  }

  if (Array.isArray(value) && value.length >= 4 && value.slice(0, 4).every((row) => Array.isArray(row))) {
    const rows = value as unknown[][];
    const rotation = rows.slice(0, 3).map((row) => row.slice(0, 3).map((item) => toNumber(item) ?? 0));
    const translation = rows.slice(0, 3).map((row) => toNumber(row[3]) ?? 0);
    return { rotation, translation };
  }

  return undefined;
};

const parseIntrinsicsEntry = (value: unknown, width?: number, height?: number) => {
  const record = asRecord(value);
  const fx = toNumber(record.fx ?? record.focal_x);
  const fy = toNumber(record.fy ?? record.focal_y);
  const cx = toNumber(record.cx ?? record.principal_x);
  const cy = toNumber(record.cy ?? record.principal_y);
  if (fx && fy && cx != null && cy != null) return { fx, fy, cx, cy };

  if (Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((row) => Array.isArray(row))) {
    const rows = value as unknown[][];
    return {
      fx: toNumber(rows[0]?.[0]) ?? 1,
      fy: toNumber(rows[1]?.[1]) ?? 1,
      cx: toNumber(rows[0]?.[2]) ?? (width ?? 2) / 2,
      cy: toNumber(rows[1]?.[2]) ?? (height ?? 2) / 2,
    };
  }

  return undefined;
};

const buildCameraTrajectoryImages = (
  cameraIntrinsics: JsonValue | undefined,
  cameraExtrinsics: JsonValue | undefined,
  fallbackImages: InputImageCameraInfo[],
): InputImageCameraInfo[] => {
  const extrinsicEntries = normalizeCameraEntries(cameraExtrinsics);
  if (!extrinsicEntries.length) return fallbackImages;

  const intrinsicEntries = normalizeCameraEntries(cameraIntrinsics);
  const fallbackByIndex = new Map(fallbackImages.map((image) => [image.index, image]));

  const parsed = extrinsicEntries
    .map((entry, index) => {
      const extrinsics = parseExtrinsicEntry(entry);
      if (!extrinsics) return undefined;
      const fallback = fallbackByIndex.get(index);
      const width = fallback?.width;
      const height = fallback?.height;
      const image: InputImageCameraInfo = {
        url: fallback?.url ?? '',
        index,
        width,
        height,
        isContextView: fallback?.isContextView ?? false,
        participatesInInference: fallback?.participatesInInference ?? false,
        cameraIntrinsics: parseIntrinsicsEntry(intrinsicEntries[index] ?? intrinsicEntries[0], width, height) ?? fallback?.cameraIntrinsics,
        cameraExtrinsics: extrinsics,
        cameraParamsStatus: 'available' as const,
      };
      return image;
    })
    .filter((item): item is InputImageCameraInfo => Boolean(item));

  return parsed.length ? parsed : fallbackImages;
};

function NovelViewCard({ hasTask, loading, error, renderedImages, gtImages, errorImages }: NovelViewCardProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('predicted');
  const hasGt = Boolean(gtImages?.length);
  const hasError = Boolean(errorImages?.length);

  const options = [
    { label: '\u9884\u6d4b\u56fe', value: 'predicted' },
    ...(hasGt ? [
      { label: '\u771f\u5b9e\u56fe', value: 'gt' },
    ] : []),
    ...(hasError ? [
      { label: '\u8bef\u5dee\u56fe', value: 'error' },
    ] : []),
  ];

  const renderImageGrid = (images: string[] | undefined, emptyText: string, altPrefix: string) => {
    if (!images?.length) return <Empty description={emptyText} />;
    return (
      <Image.PreviewGroup>
        <div className="novel-view-grid">
          {images.map((url, i) => <Image key={i} src={url} alt={`${altPrefix}-${i}`} />)}
        </div>
      </Image.PreviewGroup>
    );
  };

  const renderImages = () => {
    if (!hasTask) return <Empty description={NO_TASK_MSG} />;
    if (error) return <Empty description={'\u52a0\u8f7d\u7ed3\u679c\u5931\u8d25'} />;

    if (viewMode === 'predicted') {
      return renderImageGrid(renderedImages, '\u6682\u65e0\u9884\u6d4b\u56fe\u50cf', 'predicted');
    }

    if (viewMode === 'gt') {
      return renderImageGrid(gtImages, '\u6682\u65e0\u771f\u5b9e\u56fe\u50cf', 'gt');
    }

    return renderImageGrid(errorImages, '\u6682\u65e0\u8bef\u5dee\u56fe\u6570\u636e', 'error');
  };

  return (
    <Card
      title={'\u65b0\u89c6\u89d2\u56fe\u50cf'}
      loading={loading}
      extra={hasTask && <Segmented options={options} value={viewMode} onChange={(v) => setViewMode(v as ViewMode)} size="small" />}
    >
      {renderImages()}
    </Card>
  );
}

export function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const taskId = (location.state as { taskId?: string })?.taskId;

  const taskQuery = useTaskDetail(taskId);
  const resultQuery = useTaskResult(taskId, isTaskTerminal(taskQuery.data?.state));
  const inputImagesQuery = useTaskInputImages(taskId);

  const task = taskQuery.data;
  const result = resultQuery.data;
  const viewerUrl = result?.splatUrl ? `${VIEWER_BASE_URL}?url=${encodeURIComponent(result.splatUrl)}` : undefined;
  const hasTask = Boolean(taskId);
  const inputImages: InputImageCameraInfo[] = inputImagesQuery.data?.images ?? [];
  const trajectoryImages = buildCameraTrajectoryImages(result?.cameraIntrinsics, result?.cameraExtrinsics, inputImages);
  const contextImages = inputImages.filter((img) => img.isContextView);

  const openSplatViewer = () => {
    if (!viewerUrl) return;
    window.open(viewerUrl, '_blank', 'noopener,noreferrer');
  };

  const renderNoTask = () => <Empty description={NO_TASK_MSG} />;

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>
            {'\u4efb\u52a1\u7ed3\u679c'}
          </Typography.Title>
          {hasTask && (
            <Typography.Text type="secondary">
              {'\u4efb\u52a1 ID\uff1a'}{taskId}
              {task?.sampleName ? ` \uff5c \u6837\u672c\uff1a${stripTrailingParenthetical(task.sampleName)}` : ''}
            </Typography.Text>
          )}
        </Col>
        <Col>
          <Space>
            <Button type="primary" onClick={openSplatViewer} disabled={!viewerUrl}>
              {'Splat Viewer'}
            </Button>
            {hasTask && (
              <Button
                type="link"
                icon={<ArrowLeftOutlined />}
                onClick={() => navigate(`/tasks/${taskId}`)}
              >
                {'\u67e5\u770b\u4efb\u52a1\u8be6\u60c5'}
              </Button>
            )}
          </Space>
        </Col>
      </Row>

      <Row gutter={[16, 16]} align="stretch">
        <Col xs={24} lg={12}>
          <Space direction="vertical" size="middle" style={{ width: '100%', height: '100%' }}>
            <Card
              title={'\u8f93\u5165\u56fe\u7247\u5899'}
              loading={hasTask && inputImagesQuery.isLoading}
              className="result-left-card"
            >
              {!hasTask ? (
                renderNoTask()
              ) : (
                <InputImageGrid images={contextImages} loading={inputImagesQuery.isLoading} />
              )}
            </Card>
            <Card title={'\u76f8\u673a\u8f68\u8ff9\u53ef\u89c6\u5316'} loading={hasTask && (inputImagesQuery.isLoading || resultQuery.isLoading)} className="result-left-card">
              {!hasTask ? (
                renderNoTask()
              ) : (
                <CameraFrustumViewer images={trajectoryImages} height={260} />
              )}
            </Card>
          </Space>
        </Col>

        <Col xs={24} lg={12}>
          <Space direction="vertical" size="middle" style={{ width: '100%', height: '100%' }}>
            <Card title={'\u7ed3\u679c\u89c6\u9891'} loading={hasTask && resultQuery.isLoading} className="result-video-card">
              {!hasTask ? (
                renderNoTask()
              ) : resultQuery.isError ? (
                <Empty description={'\u52a0\u8f7d\u7ed3\u679c\u5931\u8d25'} />
              ) : result?.videoUrl ? (
                <video src={result.videoUrl} controls className="result-video-large" />
              ) : (
                <Empty description={'\u6682\u65e0\u89c6\u9891\u8f93\u51fa'} />
              )}
            </Card>
            <NovelViewCard
              hasTask={hasTask}
              loading={hasTask && resultQuery.isLoading}
              error={resultQuery.isError}
              renderedImages={result?.renderedImages}
              gtImages={result?.gtImages}
              errorImages={result?.errorImages}
            />
          </Space>
        </Col>

        <Col xs={24} lg={12}>
          <Card title={'\u6df1\u5ea6\u56fe'} loading={hasTask && resultQuery.isLoading}>
            {!hasTask ? (
              renderNoTask()
            ) : resultQuery.isError ? (
              <Empty description={'\u52a0\u8f7d\u7ed3\u679c\u5931\u8d25'} />
            ) : result?.depthImages?.length ? (
              <Image.PreviewGroup>
                <div className="depth-grid-large">
                  {result.depthImages.map((url) => (
                    <Image key={url} src={url} alt="depth" />
                  ))}
                </div>
              </Image.PreviewGroup>
            ) : (
              <Empty description={'\u6682\u65e0\u6df1\u5ea6\u56fe'} />
            )}
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title={'\u6307\u6807'} loading={hasTask && resultQuery.isLoading}>
            {!hasTask ? (
              renderNoTask()
            ) : resultQuery.isError ? (
              <Empty description={'\u52a0\u8f7d\u7ed3\u679c\u5931\u8d25'} />
            ) : result?.scores ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, minWidth: 0, overflowX: 'auto' }}>
                {result.scores.psnr != null && (
                  <Statistic title="PSNR" value={result.scores.psnr.toFixed(2)} suffix="dB" style={{ minWidth: 0 }} />
                )}
                {result.scores.ssim != null && (
                  <Statistic title="SSIM" value={result.scores.ssim.toFixed(2)} style={{ minWidth: 0 }} />
                )}
                {result.scores.lpips != null && (
                  <Statistic title="LPIPS" value={result.scores.lpips.toFixed(2)} style={{ minWidth: 0 }} />
                )}
                {result.scores.mean_pmr != null && (
                  <Statistic title="Mean PMR" value={result.scores.mean_pmr.toFixed(2)} style={{ minWidth: 0 }} />
                )}
                {result.scores.total_seconds != null && (
                  <Statistic title="耗时" value={result.scores.total_seconds.toFixed(2)} suffix="s" style={{ minWidth: 0 }} />
                )}
              </div>
            ) : result?.metrics?.length ? (
              <Row gutter={[16, 16]}>
                {result.metrics.map((metric) => (
                  <Col key={metric.label} xs={24} md={12}>
                    <Statistic title={metric.label} value={metric.value} suffix={metric.unit} />
                  </Col>
                ))}
              </Row>
            ) : (
              <Empty description={'\u6682\u65e0\u6307\u6807\u6570\u636e'} />
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
