export const STREAM_BASE_URL = import.meta.env.VITE_STREAM_BASE_URL ?? (
  import.meta.env.DEV
    ? 'http://localhost:8000'
    : (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')
)

export const BACKEND_DISABLED = import.meta.env.VITE_DISABLE_BACKEND === '1'
