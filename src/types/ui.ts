export interface SettingsState {
  backendBaseUrl: string;
  taskOptions: {
    save_video: boolean;
    save_image: boolean;
    save_gt_image: boolean;
    save_input_images: boolean;
    compute_scores: boolean;
  };
  preferences: {
    autoScrollLogs: boolean;
    compactHistory: boolean;
  };
}
