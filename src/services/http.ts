import { getSettings } from '@/store/settingsStore';

export class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

interface RequestOptions extends RequestInit {
  query?: Record<string, string | number | boolean | undefined>;
}

const buildUrl = (path: string, query?: RequestOptions['query']) => {
  const { backendBaseUrl } = getSettings();
  const url = new URL(path, backendBaseUrl.endsWith('/') ? backendBaseUrl : `${backendBaseUrl}/`);

  if (query) {
    Object.entries(query).forEach(([key, value]) => {
      if (value !== undefined) url.searchParams.set(key, String(value));
    });
  }

  return url.toString();
};

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { headers, query, ...init } = options;
  const response = await fetch(buildUrl(path, query), {
    ...init,
    headers: {
      Accept: 'application/json',
      ...headers,
    },
  });

  const contentType = response.headers.get('content-type') ?? '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === 'object' && payload && 'message' in payload
        ? String((payload as { message: unknown }).message)
        : `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status, payload);
  }

  return payload as T;
}
