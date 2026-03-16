import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { TaskCreatePage } from '@/pages/TaskCreatePage';
import { TaskDetailPage } from '@/pages/TaskDetailPage';
import { TaskHistoryPage } from '@/pages/TaskHistoryPage';
import { SettingsPage } from '@/pages/SettingsPage';

export default function AppRouter() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/tasks/new" replace />} />
        <Route path="/tasks/new" element={<TaskCreatePage />} />
        <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
        <Route path="/history" element={<TaskHistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
