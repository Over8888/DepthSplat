import { InboxOutlined } from '@ant-design/icons';
import { Alert, Upload } from 'antd';
import type { UploadFile, UploadProps } from 'antd';

interface Props {
  fileList: UploadFile[];
  onChange: (files: UploadFile[]) => void;
}

export function ImageUploader({ fileList, onChange }: Props) {
  const props: UploadProps = {
    fileList,
    multiple: true,
    beforeUpload: () => false,
    onChange: ({ fileList: next }) => onChange(next),
    accept: 'image/*',
    listType: 'picture',
  };

  return (
    <>
      <Alert
        type="info"
        showIcon
        message={'\u4e0a\u4f20\u4e00\u5f20\u6216\u591a\u5f20\u8f93\u5165\u56fe\u7247\u3002\u63d0\u4ea4\u4efb\u52a1\u524d\uff0c\u6587\u4ef6\u4ec5\u4fdd\u7559\u5728\u6d4f\u89c8\u5668\u4e2d\u3002'}
        style={{ marginBottom: 16 }}
      />
      <Upload.Dragger {...props}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">{'\u70b9\u51fb\u6216\u62d6\u62fd\u56fe\u7247\u5230\u8fd9\u91cc'}</p>
        <p className="ant-upload-hint">{'\u6d4f\u89c8\u5668\u4e0a\u4f20\u652f\u6301 PNG\u3001JPG\u3001JPEG\u3001WebP \u7b49\u5e38\u89c1\u56fe\u7247\u683c\u5f0f\u3002'}</p>
      </Upload.Dragger>
    </>
  );
}
