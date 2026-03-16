import { Typography, message } from 'antd';
import { SettingsForm } from '@/components/SettingsForm';
import { useSettings } from '@/hooks/useSettings';
import type { SettingsState } from '@/types/ui';

export function SettingsPage() {
  const { settings, updateSettings } = useSettings();

  const handleSubmit = (values: SettingsState) => {
    updateSettings(values);
    message.success('\u8bbe\u7f6e\u5df2\u4fdd\u5b58\u5230\u672c\u5730\u3002');
  };

  return (
    <>
      <Typography.Title level={3}>{'\u8bbe\u7f6e'}</Typography.Title>
      <SettingsForm initialValues={settings} onSubmit={handleSubmit} />
    </>
  );
}
