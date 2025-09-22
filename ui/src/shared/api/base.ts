const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface RequestOptions {
  headers?: Record<string, string>;
  body?: any;
}

async function request<T>(
  endpoint: string,
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET',
  options: RequestOptions = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const { headers = {}, body } = options;

  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  };

  if (body && method !== 'GET') {
    config.body = JSON.stringify(body);
  }

  const response = await fetch(url, config);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function get<T>(endpoint: string, headers?: Record<string, string>): Promise<T> {
  return request<T>(endpoint, 'GET', { headers });
}

export async function post<T>(
  endpoint: string,
  body?: any,
  headers?: Record<string, string>
): Promise<T> {
  return request<T>(endpoint, 'POST', { body, headers });
}

export async function put<T>(
  endpoint: string,
  body?: any,
  headers?: Record<string, string>
): Promise<T> {
  return request<T>(endpoint, 'PUT', { body, headers });
}

export async function postFormData<T>(
  endpoint: string,
  formData: FormData,
  baseUrl?: string
): Promise<T> {
  const url = `${baseUrl || API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'accept': 'application/json',
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}