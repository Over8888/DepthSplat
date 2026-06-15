import { InboxOutlined, UploadOutlined } from '@ant-design/icons';
import { Alert, Button, Upload } from 'antd';
import type { UploadFile, UploadProps } from 'antd';

interface Props {
  fileList: UploadFile[];
  onChange: (files: UploadFile[]) => void;
}

export function VideoUploader({ fileList, onChange }: Props) {
  const selectedFile = fileList[0];
  const selectedFileSize = selectedFile?.size ?? selectedFile?.originFileObj?.size;
  const selectedFileSizeText = selectedFileSize ? `${(selectedFileSize / 1024 / 1024).toFixed(2)} MB` : '未知大小';

  const props: UploadProps = {
    fileList,
    multiple: false,
    maxCount: 1,
    beforeUpload: () => false,
    onChange: ({ fileList: next }) => onChange(next.slice(-1)),
    accept: 'video/*',
  };

  return (
    <div className="video-uploader-panel">
      <Alert
        type="info"
        showIcon
        message={'\u9009\u62e9\u4e00\u4e2a\u8f93\u5165\u89c6\u9891\u3002\u540e\u7aef\u4f1a\u81ea\u52a8\u9009\u53d6 context \u5e76\u5904\u7406\u3002'}
        style={{ marginBottom: 16 }}
      />
      <Upload.Dragger {...props}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">{'\u70b9\u51fb\u6216\u62d6\u62fd\u89c6\u9891\u5230\u8fd9\u91cc'}</p>
        <p className="ant-upload-hint">{'\u6bcf\u6b21\u4efb\u52a1\u53ea\u652f\u6301\u4e00\u4e2a\u89c6\u9891\u6587\u4ef6\u3002'}</p>
        <Button icon={<UploadOutlined />}>{'\u9009\u62e9\u89c6\u9891'}</Button>
      </Upload.Dragger>
      {selectedFile && (
        <Alert
          type="success"
          showIcon
          message={'\u89c6\u9891\u5df2\u9009\u62e9'}
          description={`${selectedFile.name} \uff5c ${selectedFileSizeText}`}
          action={
            <Button size="small" onClick={() => onChange([])}>
              {'\u79fb\u9664'}
            </Button>
          }
          style={{ marginTop: 12 }}
        />
      )}
    </div>
  );
}
