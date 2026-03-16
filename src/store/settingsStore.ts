import { DEFAULT_PARAMETERS } from '@/utils/constants';
import type { SettingsState } from '@/types/ui';
import { readJsonStorage, writeJsonStorage } from '@/utils/storage';

const STORAGE_KEY = 'depthsplat-settings';

const defaultSettings: SettingsState = {
  backendBaseUrl: 'http://127.0.0.1:8012',
  defaultParameters: {
    numInferenceSteps: DEFAULT_PARAMETERS.numInferenceSteps,
    guidanceScale: DEFAULT_PARAMETERS.guidanceScale,
    outputFps: DEFAULT_PARAMETERS.outputFps,
    exportDepthMap: DEFAULT_PARAMETERS.exportDepthMap,
    outputFormat: DEFAULT_PARAMETERS.outputFormat,
  },
  preferences: {
    autoScrollLogs: true,
    compactHistory: false,
  },
};

export const getSettings = (): SettingsState => readJsonStorage(STORAGE_KEY, defaultSettings);
export const saveSettings = (value: SettingsState) => writeJsonStorage(STORAGE_KEY, value);
