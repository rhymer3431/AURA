import { createContext, useContext, type ReactNode } from 'react'
import { usePerceptionStreaming } from './usePerceptionStreaming'

const STREAM_BASE_URL = import.meta.env.VITE_STREAM_BASE_URL ?? 'http://localhost:8000'

type StreamingContextValue = ReturnType<typeof usePerceptionStreaming>

const StreamingContext = createContext<StreamingContextValue | null>(null)

type StreamingProviderProps = {
  baseUrl?: string
  children: ReactNode
}

export function StreamingProvider({ baseUrl = STREAM_BASE_URL, children }: StreamingProviderProps) {
  const streaming = usePerceptionStreaming({ baseUrl })
  return <StreamingContext.Provider value={streaming}>{children}</StreamingContext.Provider>
}

export function useStreaming() {
  const ctx = useContext(StreamingContext)
  if (!ctx) {
    throw new Error('useStreaming must be used within StreamingProvider')
  }
  return ctx
}
