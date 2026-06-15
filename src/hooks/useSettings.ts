import { useMemo, useSyncExternalStore } from 'react';
import type { SettingsState } from '@/types/ui';
import { getSettings, saveSettings } from '@/store/settingsStore';

let settingsState: SettingsState = getSettings();
const listeners = new Set<() => void>();

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const emitChange = () => {
  listeners.forEach((listener) => listener());
};

const getSnapshot = () => settingsState;

export function useSettings() {
  const settings = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const api = useMemo(
    () => ({
      settings,
      updateSettings(next: SettingsState) {
        settingsState = next;
        saveSettings(next);
        emitChange();
      },
    }),
    [settings],
  );

  return api;
}
