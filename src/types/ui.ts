export interface SettingsState {
  backendBaseUrl: string;
  taskOptions: {
    testChunkInterval: boolean;
    saveVideo: boolean;
    computeScores: boolean;
    exportDepthMap: boolean;
  };
  preferences: {
    autoScrollLogs: boolean;
    compactHistory: boolean;
  };
}
