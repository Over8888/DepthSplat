import type { SettingsState } from '@/types/ui';
import { readJsonStorage, writeJsonStorage } from '@/utils/storage';

const STORAGE_KEY = 'depthsplat-settings';

const defaultSettings: SettingsState = {
  backendBaseUrl: 'http://127.0.0.1:8012',
  taskOptions: {
    testChunkInterval: true,
    saveVideo: true,
    computeScores: false,
    exportDepthMap: true,
  },
  preferences: {
    autoScrollLogs: true,
    compactHistory: false,
  },
};

export const getSettings = (): SettingsState => {
  const stored = readJsonStorage<Partial<SettingsState>>(STORAGE_KEY, defaultSettings);

  return {
    backendBaseUrl: stored.backendBaseUrl ?? defaultSettings.backendBaseUrl,
    taskOptions: {
      ...defaultSettings.taskOptions,
      ...(stored.taskOptions ?? {}),
    },
    preferences: {
      ...defaultSettings.preferences,
      ...(stored.preferences ?? {}),
    },
  };
};

export const saveSettings = (value: SettingsState) => writeJsonStorage(STORAGE_KEY, value);
