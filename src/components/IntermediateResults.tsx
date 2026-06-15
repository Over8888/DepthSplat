import JSZip from 'jszip';
import { Button, Card, Collapse, Descriptions, Empty, Image, Space, Table, Typography, message } from 'antd';
import { DownloadOutlined, DownOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { CameraFrustumViewer } from '@/components/CameraFrustumViewer';
import { InputImageGrid } from '@/components/InputImageGrid';
import type { InputImageCameraInfo, JsonValue, TaskDetail, TaskResult } from '@/types/api';

const getFileName = (url: string): string => {
  try {
    const pathname = new URL(url).pathname;
    return pathname.split('/').pop() || 'file';
  } catch {
    return url.split('/').pop() || 'file';
  }
};

const triggerBlobDownload = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

const downloadFile = async (url: string, filename?: string) => {
  const name = filename || getFileName(url);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Download failed');
    const blob = await response.blob();
    triggerBlobDownload(blob, name);
    message.success('\u4e0b\u8f7d\u5b8c\u6210');
  } catch {
    window.open(url, '_blank', 'noopener,noreferrer');
  }
};

const downloadJSON = (data: unknown, filename: string) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  triggerBlobDownload(blob, filename);
  message.success('\u4e0b\u8f7d\u5b8c\u6210');
};

const formatMatrixValue = (value: JsonValue | undefined): string => {
  if (value === undefined) return '-';
  if (value === null || typeof value !== 'object') return String(value);
  return JSON.stringify(value);
};

const formatVectorValue = (value: JsonValue | undefined): string => {
  if (value === undefined) return '-';
  if (Array.isArray(value)) {
    return `[${value.map(v => typeof v === 'number' ? v.toFixed(4) : v).join(', ')}]`;
  }
  return formatMatrixValue(value);
};

const isMatrixRows = (value: JsonValue | undefined): value is JsonValue[] =>
  Array.isArray(value) && value.every((row) => Array.isArray(row));

const NO_DATA_TEXT = '\u672a\u67e5\u8be2\u5230\u76f8\u5173\u6570\u636e';

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <Typography.Title level={5} style={{ margin: '0 0 8px 0' }}>{children}</Typography.Title>
);

const parseJsonValue = (v: JsonValue): JsonValue => {
  if (typeof v === 'string') {
    try { return JSON.parse(v); } catch { return v; }
  }
  return v;
};

const IntrinsicsViewer = ({ value, filename }: { value?: JsonValue; filename: string }) => {
  if (value === undefined) {
    return (
      <div>
        <SectionTitle>{'\u76f8\u673a\u5185\u53c2\u77e9\u9635'}</SectionTitle>
        <Empty description={NO_DATA_TEXT} />
      </div>
    );
  }

  const raw = parseJsonValue(value);
  const items = Array.isArray(raw) ? raw.map(parseJsonValue) : [raw];
  const first = items[0];
  const isObj = first !== null && typeof first === 'object' && !Array.isArray(first);

  return (
    <div>
      <Space align="center" style={{ marginBottom: 8 }}>
        <SectionTitle>{'\u76f8\u673a\u5185\u53c2\u77e9\u9635'}</SectionTitle>
        <Button size="small" icon={<DownloadOutlined />} onClick={() => downloadJSON(value, filename)}>
          {'\u5bfc\u51fa JSON'}
        </Button>
      </Space>
      {isObj ? (
        <Descriptions bordered size="small" column={{ xs: 2, sm: 4 }}>
          {Object.entries(first as Record<string, JsonValue>).map(([key, val]) => (
            <Descriptions.Item key={key} label={key}>
              {typeof val === 'number' ? val.toFixed(4) : formatMatrixValue(val)}
            </Descriptions.Item>
          ))}
        </Descriptions>
      ) : isMatrixRows(first) ? (
        <Table
          size="small"
          bordered
          pagination={false}
          dataSource={(first as JsonValue[][]).map((row, idx) => ({ key: idx, row }))}
          columns={Array.from({ length: (first as JsonValue[][])[0]?.length ?? 0 }, (_, ci) => ({
            title: `C${ci + 1}`,
            dataIndex: 'row',
            key: String(ci),
            render: (row: JsonValue[]) => typeof row[ci] === 'number' ? (row[ci] as number).toFixed(4) : formatMatrixValue(row[ci]),
          }))}
        />
      ) : (
        <Table
          size="small"
          bordered
          pagination={false}
          dataSource={[{ key: 0, values: items }]}
          columns={Array.from({ length: items.length }, (_, index) => ({
            title: `C${index + 1}`,
            dataIndex: 'values',
            key: String(index),
            render: (values: JsonValue[]) => formatMatrixValue(values[index]),
          }))}
        />
      )}
      {items.length > 1 && isObj && (
        <Typography.Text type="secondary" style={{ fontSize: 12, marginTop: 4, display: 'block' }}>
          {'\u5171 '}{items.length}{' \u7ec4\u5185\u53c2\uff0c\u4ec5\u663e\u793a\u7b2c\u4e00\u7ec4'}
        </Typography.Text>
      )}
    </div>
  );
};

const ExtrinsicsViewer = ({ value, filename }: { value?: JsonValue; filename: string }) => {
  const [expanded, setExpanded] = useState(false);

  if (value === undefined) {
    return (
      <div>
        <SectionTitle>{'\u76f8\u673a\u5916\u53c2\u77e9\u9635'}</SectionTitle>
        <Empty description={NO_DATA_TEXT} />
      </div>
    );
  }

  const raw = parseJsonValue(value);
  const items = Array.isArray(raw) ? raw.map(parseJsonValue) : [raw];

  type ExtrinsicEntry = { rotation: number[]; translation: number[] };
  const isObjArray = items.length > 0 && items[0] !== null && typeof items[0] === 'object' && !Array.isArray(items[0]) && ('rotation' in (items[0] as object) || 'translation' in (items[0] as object));

  const entries: ExtrinsicEntry[] = isObjArray
    ? (items as unknown as ExtrinsicEntry[])
    : [];

  const is4x4Matrix = !isObjArray && items.length > 0 && isMatrixRows(items[0]) && (items[0] as JsonValue[]).length === 4;
  const isVectorArray = !isObjArray && !is4x4Matrix && items.length > 0 && Array.isArray(items[0]);

  const previewCount = 3;
  const displayEntries = expanded ? entries : entries.slice(0, previewCount);
  const displayItems = expanded ? items : items.slice(0, previewCount);
  const displayVectors = expanded ? items : items.slice(0, previewCount);

  return (
    <div>
      <Space align="center" style={{ marginBottom: 8 }}>
        <SectionTitle>{'\u76f8\u673a\u5916\u53c2\u77e9\u9635'}</SectionTitle>
        <Button size="small" icon={<DownloadOutlined />} onClick={() => downloadJSON(value, filename)}>
          {'\u5bfc\u51fa JSON'}
        </Button>
      </Space>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {isObjArray ? (
          displayEntries.map((entry, idx) => (
            <div key={idx} style={{ padding: '8px 12px', border: '1px solid #d9d9d9', borderRadius: 4, backgroundColor: '#fafafa' }}>
              <div style={{ marginBottom: 4 }}>
                <Typography.Text strong style={{ fontSize: 12, color: '#666' }}>
                  Rotation {idx + 1}
                </Typography.Text>
                <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                  {formatVectorValue(entry.rotation)}
                </div>
              </div>
              <div>
                <Typography.Text strong style={{ fontSize: 12, color: '#666' }}>
                  Translation {idx + 1}
                </Typography.Text>
                <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                  {formatVectorValue(entry.translation)}
                </div>
              </div>
            </div>
          ))
        ) : is4x4Matrix ? (
          displayItems.map((mat, idx) => (
            <div key={idx} style={{ padding: '8px 12px', border: '1px solid #d9d9d9', borderRadius: 4, backgroundColor: '#fafafa' }}>
              <Typography.Text strong style={{ fontSize: 12, color: '#666' }}>
                {'外参矩阵 '}{idx + 1}
              </Typography.Text>
              <Table
                size="small"
                bordered
                pagination={false}
                style={{ marginTop: 4 }}
                dataSource={(mat as JsonValue[][]).map((row, ri) => ({ key: ri, row }))}
                columns={Array.from({ length: 4 }, (_, ci) => ({
                  title: `C${ci + 1}`,
                  dataIndex: 'row',
                  key: String(ci),
                  render: (row: JsonValue[]) => typeof row[ci] === 'number' ? (row[ci] as number).toFixed(6) : formatMatrixValue(row[ci]),
                }))}
              />
            </div>
          ))
        ) : isVectorArray ? (
          (displayVectors as JsonValue[][]).map((row, idx) => (
            <div key={idx} style={{ padding: '8px 12px', border: '1px solid #d9d9d9', borderRadius: 4, backgroundColor: '#fafafa' }}>
              <Typography.Text strong style={{ fontSize: 12, color: '#666' }}>
                {'向量 '}{idx + 1}
              </Typography.Text>
              <div style={{ marginTop: 4, fontFamily: 'monospace', fontSize: 13 }}>
                {formatVectorValue(row)}
              </div>
            </div>
          ))
        ) : (
          <div style={{ padding: '8px 12px', border: '1px solid #d9d9d9', borderRadius: 4, backgroundColor: '#fafafa' }}>
            <pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>{JSON.stringify(value, null, 2)}</pre>
          </div>
        )}
      </div>
      {items.length > previewCount && (
        <Button
          type="link"
          size="small"
          icon={<DownOutlined style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />}
          onClick={() => setExpanded(!expanded)}
          style={{ marginTop: 8, padding: 0 }}
        >
          {expanded ? '\u6536\u8d77' : `\u5c55\u5f00\u5168\u90e8 ${items.length} \u7ec4`}
        </Button>
      )}
    </div>
  );
};

const ImageAssetSection = ({ title, images, folder }: { title: string; images: string[]; folder: string }) => {
  return (
    <div>
      <SectionTitle>{images.length ? `${title}\uff08${images.length}\uff09` : title}</SectionTitle>
      {images.length ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
          <Image.PreviewGroup>
            {images.map((url, index) => (
              <div key={`${folder}-${url}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                <Image src={url} alt="" width={160} height={120} style={{ objectFit: 'cover', borderRadius: 8 }} />
                <Button size="small" type="link" icon={<DownloadOutlined />} onClick={() => downloadFile(url, `${folder}-${String(index + 1).padStart(3, '0')}-${getFileName(url)}`)}>
                  {'\u4e0b\u8f7d'}
                </Button>
              </div>
            ))}
          </Image.PreviewGroup>
        </div>
      ) : (
        <Empty description={NO_DATA_TEXT} />
      )}
    </div>
  );
};

interface Props {
  task?: TaskDetail;
  result?: TaskResult;
  loading?: boolean;
}

export function IntermediateResults({ task, result, loading }: Props) {
  const inputImages = result?.inputImages ?? task?.inputImages ?? [];
  const renderedImages = result?.renderedImages ?? task?.renderedImages ?? result?.previewImages ?? [];
  const gtImages = result?.gtImages ?? task?.gtImages ?? [];
  const splatUrl = result?.splatUrl;
  const parameters = result?.parameters ?? task?.parameters;
  const cameraIntrinsics = result?.cameraIntrinsics ?? task?.cameraIntrinsics ?? parameters?.cameraIntrinsics ?? parameters?.camera_intrinsics ?? parameters?.intrinsics ?? parameters?.K ?? parameters?.k;
  const cameraExtrinsics = result?.cameraExtrinsics ?? task?.cameraExtrinsics ?? parameters?.cameraExtrinsics ?? parameters?.camera_extrinsics ?? parameters?.extrinsics ?? parameters?.cameraToWorld ?? parameters?.camera_to_world ?? parameters?.c2w;

  const hasAny =
    inputImages.length > 0 ||
    renderedImages.length > 0 ||
    gtImages.length > 0 ||
    Boolean(splatUrl) ||
    cameraIntrinsics !== undefined ||
    cameraExtrinsics !== undefined ||
    (parameters && Object.keys(parameters).length > 0);

  const addRemoteFilesToZip = async (zip: JSZip, folder: string, urls: string[], failedFiles: string[]) => {
    await Promise.all(
      urls.map(async (url, index) => {
        try {
          const response = await fetch(url);
          if (!response.ok) throw new Error('Download failed');
          zip.file(`${folder}/${String(index + 1).padStart(3, '0')}-${getFileName(url)}`, await response.blob());
        } catch {
          failedFiles.push(url);
        }
      }),
    );
  };

  const handleDownloadAll = async () => {
    const zip = new JSZip();
    const failedFiles: string[] = [];

    await addRemoteFilesToZip(zip, 'input-images', inputImages, failedFiles);
    await addRemoteFilesToZip(zip, 'rendered-images', renderedImages, failedFiles);
    await addRemoteFilesToZip(zip, 'gt-images', gtImages, failedFiles);

    if (splatUrl) {
      try {
        const response = await fetch(splatUrl);
        if (!response.ok) throw new Error('Download failed');
        zip.file(`gaussian/${getFileName(splatUrl)}`, await response.blob());
      } catch {
        failedFiles.push(splatUrl);
      }
    }

    if (parameters) zip.file('parameters.json', JSON.stringify(parameters, null, 2));
    if (cameraIntrinsics !== undefined) zip.file('camera-intrinsics.json', JSON.stringify(cameraIntrinsics, null, 2));
    if (cameraExtrinsics !== undefined) zip.file('camera-extrinsics.json', JSON.stringify(cameraExtrinsics, null, 2));
    if (failedFiles.length) zip.file('download-failures.json', JSON.stringify(failedFiles, null, 2));

    const blob = await zip.generateAsync({ type: 'blob' });
    triggerBlobDownload(blob, `task-${result?.taskId ?? task?.id ?? 'results'}-assets.zip`);
    message.success(failedFiles.length ? '\u90e8\u5206\u6587\u4ef6\u6253\u5305\u5b8c\u6210' : '\u4e00\u952e\u6253\u5305\u5b8c\u6210');
  };

  return (
    <Card
      title={'\u4efb\u52a1\u4ea7\u7269'}
      loading={loading}
      extra={hasAny ? <Button icon={<DownloadOutlined />} onClick={handleDownloadAll}>{'\u4e00\u952e\u6253\u5305\u4e0b\u8f7d'}</Button> : null}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <ImageAssetSection title={'\u8f93\u5165\u56fe\u50cf'} images={inputImages} folder="input-image" />
        <IntrinsicsViewer value={cameraIntrinsics} filename="camera-intrinsics.json" />
        <ExtrinsicsViewer value={cameraExtrinsics} filename="camera-extrinsics.json" />
        <div>
          <SectionTitle>{'\u4efb\u52a1\u53c2\u6570'}</SectionTitle>
          {parameters && Object.keys(parameters).length > 0 ? (
            <>
              <Descriptions bordered size="small" column={{ xs: 1, sm: 2, lg: 3 }}>
                {Object.entries(parameters).map(([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    {formatMatrixValue(value)}
                  </Descriptions.Item>
                ))}
              </Descriptions>
              <Button size="small" icon={<DownloadOutlined />} style={{ marginTop: 8 }} onClick={() => downloadJSON(parameters, 'parameters.json')}>
                {'\u5bfc\u51fa JSON'}
              </Button>
            </>
          ) : (
            <Empty description={NO_DATA_TEXT} />
          )}
        </div>
        <ImageAssetSection title={'\u65b0\u89c6\u89d2\u6e32\u67d3\u56fe\u50cf'} images={renderedImages} folder="rendered-image" />
        <ImageAssetSection title={'GT \u56fe\u50cf'} images={gtImages} folder="gt-image" />
        <div>
          <SectionTitle>{'\u9ad8\u65af\u8868\u793a\uff08.ply\uff09'}</SectionTitle>
          {splatUrl ? (
            <Button icon={<DownloadOutlined />} onClick={() => downloadFile(splatUrl)}>
              {'\u4e0b\u8f7d '}{getFileName(splatUrl)}
            </Button>
          ) : (
            <Empty description={NO_DATA_TEXT} />
          )}
        </div>
      </Space>
    </Card>
  );
}