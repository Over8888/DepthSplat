import { Button, Modal, Space } from 'antd';
import type { BackendTaskState } from '@/types/api';
import { isTaskCancellable } from '@/utils/task';

interface Props {
  state?: BackendTaskState;
  cancelling?: boolean;
  onCancel: () => void | Promise<void>;
}

export function TaskControl({ state, cancelling, onCancel }: Props) {
  const handleClick = () => {
    Modal.confirm({
      title: '\u786e\u8ba4\u53d6\u6d88\u8be5\u4efb\u52a1\u5417\uff1f',
      content: '\u754c\u9762\u4f1a\u6301\u7eed\u8f6e\u8be2\uff0c\u76f4\u5230\u540e\u7aef\u8fd4\u56de\u4efb\u52a1\u72b6\u6001\u4e3a\u5df2\u53d6\u6d88\u3002',
      okText: '\u53d6\u6d88\u4efb\u52a1',
      okButtonProps: { danger: true },
      onOk: onCancel,
    });
  };

  if (!isTaskCancellable(state)) return null;

  return (
    <Space>
      <Button danger loading={cancelling} onClick={handleClick}>
        {cancelling ? '\u6b63\u5728\u53d6\u6d88\u2026' : '\u53d6\u6d88\u4efb\u52a1'}
      </Button>
    </Space>
  );
}
