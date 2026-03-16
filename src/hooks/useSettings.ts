import { useMemo, useState } from 'react';
import type { SettingsState } from '@/types/ui';
import { getSettings, saveSettings } from '@/store/settingsStore';

export function useSettings() {
  const [settings, setSettings] = useState<SettingsState>(() => getSettings());

  const api = useMemo(
    () => ({
      settings,
      updateSettings(next: SettingsState) {
        saveSettings(next);
        setSettings(next);
      },
    }),
    [settings],
  );

  return api;
}
