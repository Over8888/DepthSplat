import { Empty, Image, Tooltip } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, WarningOutlined } from '@ant-design/icons';
import type { InputImageCameraInfo } from '@/types/api';

interface Props {
  images?: InputImageCameraInfo[];
  loading?: boolean;
}

const statusMeta = (status: InputImageCameraInfo['cameraParamsStatus']) => {
  switch (status) {
    case 'available':
      return { icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />, text: '已提供相机参数', color: '#52c41a' };
    case 'partial':
      return { icon: <WarningOutlined style={{ color: '#faad14' }} />, text: '相机参数不完整', color: '#faad14' };
    case 'missing':
      return { icon: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />, text: '未提供相机参数', color: '#ff4d4f' };
  }
};

export function InputImageGrid({ images, loading }: Props) {
  if (!images?.length) {
    if (loading) return null;
    return <Empty description={'\u6682\u65e0\u8f93\u5165\u56fe\u7247\u6570\u636e'} />;
  }

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
      <Image.PreviewGroup>
        {images.map((img) => {
          const meta = statusMeta(img.cameraParamsStatus);
          const resolution = img.width && img.height ? `${img.width}×${img.height}` : '未知分辨率';

          return (
            <div
              key={img.url}
              style={{
                position: 'relative',
                border: img.isContextView ? '3px solid #1677ff' : '1px solid #d9d9d9',
                borderRadius: 8,
                overflow: 'hidden',
                width: 180,
                opacity: img.participatesInInference ? 1 : 0.4,
                boxShadow: img.isContextView ? '0 0 0 2px rgba(22,119,255,0.15)' : 'none',
              }}
            >
              <Image src={img.url} alt={`img-${img.index}`} width={180} height={135} style={{ objectFit: 'cover', borderRadius: 0 }} />
              <div
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  padding: '6px 8px',
                  background: 'linear-gradient(to top, rgba(0,0,0,0.7), rgba(0,0,0,0))',
                  color: '#fff',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, whiteSpace: 'nowrap' }}>{resolution}</span>
                  <Tooltip title={meta.text}>
                    <span>{meta.icon}</span>
                  </Tooltip>
                </div>
              </div>
            </div>
          );
        })}
      </Image.PreviewGroup>
    </div>
  );
}
