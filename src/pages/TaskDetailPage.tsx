import { Alert, Col, Row, Space, Typography, message } from 'antd';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { CancelBanner } from '@/components/CancelBanner';
import { IntermediateResults } from '@/components/IntermediateResults';
import { LogViewer } from '@/components/LogViewer';
import { TaskControl } from '@/components/TaskControl';
import { TaskStatusCard } from '@/components/TaskStatusCard';
import { useCancelTask, useTaskDetail, useTaskLogs, useTaskResult } from '@/hooks/useTask';
import { buildErrorSummary, isCancelledTask, isFailedTask, isTaskTerminal } from '@/utils/task';

export function TaskDetailPage() {
  const { taskId = '' } = useParams();
  const [localState, setLocalState] = useState<string>();
  const taskQuery = useTaskDetail(taskId);
  const cancelTask = useCancelTask(taskId);
  const task = taskQuery.data;
  const logsQuery = useTaskLogs(taskId, task?.state, Boolean(taskId));
  const resultQuery = useTaskResult(taskId, isTaskTerminal(task?.state));

  useEffect(() => {
    if (task?.state === 'cancelled') {
      setLocalState(undefined);
    } else if (cancelTask.isPending) {
      setLocalState('cancelling');
    }
  }, [task?.state, cancelTask.isPending]);

  const handleCancel = async () => {
    try {
      setLocalState('cancelling');
      await cancelTask.mutateAsync();
      message.info('\u5df2\u53d1\u9001\u53d6\u6d88\u8bf7\u6c42\uff0c\u7b49\u5f85\u540e\u7aef\u786e\u8ba4\u3002');
      await taskQuery.refetch();
    } catch (error) {
      setLocalState(undefined);
      message.error(error instanceof Error ? error.message : '\u53d6\u6d88\u4efb\u52a1\u5931\u8d25');
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>
            {'\u4efb\u52a1\u8be6\u60c5'}
          </Typography.Title>
          <Typography.Text type="secondary">{`\u4efb\u52a1 ID\uff1a${taskId}`}</Typography.Text>
        </Col>
        <Col>
          <TaskControl state={task?.state} cancelling={localState === 'cancelling'} onCancel={handleCancel} />
        </Col>
      </Row>

      {isCancelledTask(task?.state) && <CancelBanner />}
      {taskQuery.isError && <Alert type="error" showIcon message={'\u52a0\u8f7d\u4efb\u52a1\u8be6\u60c5\u5931\u8d25'} description={taskQuery.error instanceof Error ? taskQuery.error.message : '\u672a\u77e5\u9519\u8bef'} />}

      <Row gutter={[16, 16]} align="stretch">
        <Col xs={24} xl={8}>
          <TaskStatusCard task={task} loading={taskQuery.isLoading} localState={localState} />
        </Col>
        <Col xs={24} xl={16}>
          <LogViewer entries={logsQuery.data?.entries} timings={task?.timings} loading={logsQuery.isLoading} refreshing={logsQuery.isFetching && !logsQuery.isLoading} />
        </Col>
      </Row>

      {resultQuery.data && (
        <IntermediateResults task={task} result={resultQuery.data} loading={resultQuery.isLoading} />
      )}
      {isFailedTask(task?.state) && buildErrorSummary(task) && <Typography.Text type="danger">{buildErrorSummary(task)}</Typography.Text>}
    </Space>
  );
}
