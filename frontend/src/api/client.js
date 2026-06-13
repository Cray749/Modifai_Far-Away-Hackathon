import ky from 'ky'

// ─── Base API Client ──────────────────────────────────────────────────────────
// All HTTP requests go through this configured ky instance.
// Update prefixUrl when the backend URL changes (e.g. staging, production).

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'

export const api = ky.create({
    prefixUrl: API_BASE,
    timeout: 30_000,
    retry: {
        limit: 2,
        methods: ['get'],          // only retry idempotent requests
        statusCodes: [408, 502, 503, 504],
    },
})


// ─── SSE (Server-Sent Events) ─────────────────────────────────────────────────
// Use for real-time pipeline progress streaming.
//
// Usage:
//   const unsubscribe = subscribeToStream('/projects/proj-001/stream', {
//       onMessage: (data) => console.log(data),
//       onError:   (err)  => console.error(err),
//   })
//   // Later: unsubscribe()

export function subscribeToStream(path, { onMessage, onError, onOpen } = {}) {
    const url = `${API_BASE}/${path.replace(/^\//, '')}`
    const source = new EventSource(url)

    if (onOpen) {
        source.addEventListener('open', onOpen)
    }

    source.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data)
            onMessage?.(data)
        } catch {
            // Non-JSON message, pass raw data
            onMessage?.(event.data)
        }
    }

    source.onerror = (event) => {
        onError?.(event)
        // EventSource auto-reconnects on error; close if needed
    }

    // Return cleanup function
    return () => {
        source.close()
    }
}


// ─── Convenience Helpers ──────────────────────────────────────────────────────
// Typed wrappers around the ky instance for common patterns.

import { mockState } from './mockState'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS !== 'false' // Default to true for frontend testing

export const apiClient = {
    get: async (path, options) => {
        if (USE_MOCKS) {
            const cleanPath = path.split('?')[0]
            await new Promise(r => setTimeout(r, 200)) // network delay
            if (cleanPath === 'projects') return mockState.getProjects()
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/status')) {
                const id = cleanPath.split('/')[1]
                return { status: mockState.getProject(id).status }
            }
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/logs')) {
                const id = cleanPath.split('/')[1]
                return { logs: mockState.getProject(id).logs }
            }
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/results')) {
                const id = cleanPath.split('/')[1]
                return mockState.getProject(id).results
            }
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/dataset')) {
                return { items: [{ content: "Mock dataset item 1" }, { content: "Mock dataset item 2" }] }
            }
            if (cleanPath.startsWith('projects/') && cleanPath.includes('/dataset/search')) {
                return { items: [{ content: "Mock search result" }] }
            }
            if (cleanPath.startsWith('projects/')) {
                const id = cleanPath.split('/')[1]
                return mockState.getProject(id)
            }
        }
        return api.get(path, options).json()
    },
    
    post: async (path, body, options) => {
        if (USE_MOCKS) {
            const cleanPath = path.split('?')[0]
            await new Promise(r => setTimeout(r, 400))
            if (cleanPath === 'evaluate') {
                return { score: 0.95, explanation: 'Mocked evaluation: High Quality document suitable for the pipeline.' }
            }
            if (cleanPath === 'projects') {
                return mockState.createProject(body)
            }
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/upload-url')) {
                return { presigned_url: '/mock-upload-url' }
            }
            if (cleanPath.startsWith('projects/') && cleanPath.endsWith('/start')) {
                const id = cleanPath.split('/')[1]
                return mockState.startPipeline(id)
            }
        }
        return api.post(path, { json: body, ...options }).json()
    },
    
    put: async (path, body, options) => {
        if (USE_MOCKS) {
            await new Promise(r => setTimeout(r, 500))
            return { status: 'uploaded' }
        }
        return api.put(path, { json: body, ...options }).json()
    },
    
    patch: (path, body, options) => api.patch(path, { json: body, ...options }).json(),
    
    delete: async (path, options) => {
        if (USE_MOCKS) {
            const cleanPath = path.split('?')[0]
            if (cleanPath.startsWith('projects/')) {
                const id = cleanPath.split('/')[1]
                return mockState.deleteProject(id)
            }
        }
        return api.delete(path, options).json()
    },

    upload: (path, formData, options) =>
        api.post(path, { body: formData, ...options }).json(),
}

export default apiClient
