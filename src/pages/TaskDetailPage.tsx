import { Alert, Card, Col, Empty, Row, Space, Typography, message } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { CancelBanner } from '@/components/CancelBanner';
import { LogViewer } from '@/components/LogViewer';
import { ResultViewer } from '@/components/ResultViewer';
import { TaskControl } from '@/components/TaskControl';
import { TaskStatusCard } from '@/components/TaskStatusCard';
import { useCancelTask, useTaskDetail, useTaskLogs, useTaskResult } from '@/hooks/useTask';
import { buildErrorSummary, canShowPartialResults, isCancelledTask, isFailedTask, isSuccessTask, isTaskTerminal } from '@/utils/task';

export function TaskDetailPage() {
  const { taskId = '' } = useParams();
  const [localState, setLocalState] = useState<string>();
  const taskQuery = useTaskDetail(taskId);
  const cancelTask = useCancelTask(taskId);
  const task = taskQuery.data;
  const logsQuery = useTaskLogs(taskId, Boolean(taskId));
  const resultQuery = useTaskResult(taskId, isTaskTerminal(task?.state));

  useEffect(() => {
    if (task?.state === 'cancelled') {
      setLocalState(undefined);
    } else if (cancelTask.isPending) {
      setLocalState('cancelling');
    }
  }, [task?.state, cancelTask.isPending]);

  const resultMode = useMemo(() => {
    if (isSuccessTask(task?.state)) return 'success' as const;
    if (isFailedTask(task?.state)) return 'failed' as const;
    return 'cancelled' as const;
  }, [task?.state]);

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

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={8}>
          <TaskStatusCard task={task} loading={taskQuery.isLoading} localState={localState} />
        </Col>
        <Col xs={24} xl={16}>
          <LogViewer entries={logsQuery.data?.entries} loading={logsQuery.isLoading || logsQuery.isFetching} />
        </Col>
      </Row>

      {isSuccessTask(task?.state) && <ResultViewer task={task} result={resultQuery.data} loading={resultQuery.isLoading} mode={resultMode} />}
      {isFailedTask(task?.state) && <ResultViewer task={task} result={resultQuery.data} loading={resultQuery.isLoading} mode={resultMode} />}
      {isCancelledTask(task?.state) && canShowPartialResults(task, resultQuery.data) && (
        <ResultViewer task={task} result={resultQuery.data} loading={resultQuery.isLoading} mode={resultMode} />
      )}
      {isCancelledTask(task?.state) && !canShowPartialResults(task, resultQuery.data) && (
        <Card title={'\u90e8\u5206\u7ed3\u679c'}>
          <Empty description={'\u8be5\u5df2\u53d6\u6d88\u4efb\u52a1\u6682\u65e0\u540e\u7aef\u8fd4\u56de\u7684\u90e8\u5206\u7ed3\u679c\u3002'} />
        </Card>
      )}
      {isFailedTask(task?.state) && buildErrorSummary(task) && <Typography.Text type="danger">{buildErrorSummary(task)}</Typography.Text>}
    </Space>
  );
}
