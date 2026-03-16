export interface SettingsState {
  backendBaseUrl: string;
  defaultParameters: {
    numInferenceSteps: number;
    guidanceScale: number;
    outputFps: number;
    exportDepthMap: boolean;
    outputFormat: 'mp4' | 'webm';
  };
  preferences: {
    autoScrollLogs: boolean;
    compactHistory: boolean;
  };
}
