import { useSyncExternalStore } from 'react';

type CurrentTaskState = {
  taskId?: string;
  presetId?: string;
  sampleId?: string;
};

let state: CurrentTaskState = {};

const listeners = new Set<() => void>();

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const emitChange = () => {
  listeners.forEach((listener) => listener());
};

export const getCurrentTaskState = () => state;

export const setCurrentTaskState = (next: Partial<CurrentTaskState>) => {
  state = { ...state, ...next };
  emitChange();
};

export const setCurrentTaskId = (taskId?: string) => {
  setCurrentTaskState({ taskId });
};

export const useCurrentTask = () => useSyncExternalStore(subscribe, getCurrentTaskState, getCurrentTaskState);
