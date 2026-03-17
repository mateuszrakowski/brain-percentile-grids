const BASE_URL = '';

let authToken = localStorage.getItem('auth_token');

export function setToken(token) {
    authToken = token;
    if (token) {
        localStorage.setItem('auth_token', token);
    } else {
        localStorage.removeItem('auth_token');
    }
}

export function getToken() {
    return authToken;
}

export function isAuthenticated() {
    return !!authToken;
}

async function request(method, path, { body, formData, query, stream } = {}) {
    const url = new URL(path, window.location.origin);
    if (query) {
        for (const [k, v] of Object.entries(query)) {
            if (v != null) {
                if (Array.isArray(v)) {
                    v.forEach(val => url.searchParams.append(k, val));
                } else {
                    url.searchParams.set(k, v);
                }
            }
        }
    }

    const headers = {};
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }

    let fetchBody;
    if (formData) {
        fetchBody = formData;
    } else if (body) {
        headers['Content-Type'] = 'application/json';
        fetchBody = JSON.stringify(body);
    }

    const res = await fetch(url.toString(), { method, headers, body: fetchBody });

    if (stream) {
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new ApiError(res.status, err.detail || err.error || 'Request failed');
        }
        return res;
    }

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new ApiError(res.status, err.detail || err.error || 'Request failed');
    }

    return res.json();
}

export class ApiError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
    }
}

// Auth
export const auth = {
    register: (username, password) =>
        request('POST', '/api/auth/register', { body: { username, password } }),
    login: async (username, password) => {
        const form = new URLSearchParams();
        form.append('username', username);
        form.append('password', password);
        const res = await fetch(`${BASE_URL}/api/auth/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new ApiError(res.status, err.detail || 'Login failed');
        }
        return res.json();
    },
    me: () => request('GET', '/api/auth/me'),
};

// Datasets
export const datasets = {
    list: () => request('GET', '/api/datasets'),
    get: (id) => request('GET', `/api/datasets/${id}`),
    create: (name, description) =>
        request('POST', '/api/datasets', { body: { name, description } }),
    update: (id, data) => request('PATCH', `/api/datasets/${id}`, { body: data }),
    delete: (id) => request('DELETE', `/api/datasets/${id}`),
};

// Data
export const data = {
    upload: (datasetId, files) => {
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        return request('POST', `/api/datasets/${datasetId}/upload`, { formData: fd });
    },
    get: (datasetId) => request('GET', `/api/datasets/${datasetId}/data`),
    getTable: (datasetId) => request('GET', `/api/datasets/${datasetId}/data/table`),
    clear: (datasetId) => request('DELETE', `/api/datasets/${datasetId}/data`),
    structures: (datasetId) => request('GET', `/api/datasets/${datasetId}/structures`),
};

// Calculations
export const calculations = {
    fit: (datasetId, body) =>
        request('POST', `/api/datasets/${datasetId}/fit`, { body }),
    fitStream: (datasetId, body) =>
        request('POST', `/api/datasets/${datasetId}/fit/stream`, { body, stream: true }),
    calculate: (datasetId, files, structures) => {
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        return request('POST', `/api/datasets/${datasetId}/calculate`, {
            formData: fd,
            query: structures ? { structures } : undefined,
        });
    },
    listCalculations: (datasetId, includeStale = false) =>
        request('GET', `/api/datasets/${datasetId}/calculations`, {
            query: includeStale ? { include_stale: true } : undefined,
        }),
    getCalculation: (datasetId, calculationId) =>
        request('GET', `/api/datasets/${datasetId}/calculations/${calculationId}`),
    getResultPlotUrl: (datasetId, calculationId, resultId) =>
        `/api/datasets/${datasetId}/calculations/${calculationId}/results/${resultId}/plot`,
    getReferencePlotUrl: (datasetId, structure) =>
        `/api/datasets/${datasetId}/models/${encodeURIComponent(structure)}/reference-plot`,
    deleteCalculation: (datasetId, calculationId) =>
        request('DELETE', `/api/datasets/${datasetId}/calculations/${calculationId}`),
};
