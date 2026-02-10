import { useEffect, useMemo, useRef, useState } from 'react'
import type { StreamMetadata, StreamState } from '../../domain/streaming/streamTypes'
import { EMPTY_METADATA, parseMetadataMessage } from '../../infra/streaming/metadataParsers'
import { WebRtcClient } from '../../infra/streaming/webRtcClient'
import { WebSocketClient } from '../../infra/streaming/webSocketClient'

type HookArgs = {
  baseUrl: string
  wsUrl?: string
}

export function usePerceptionStreaming({ baseUrl }: HookArgs) {
  const [streamState, setStreamState] = useState<StreamState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState<number>(0)
  const [hasVideo, setHasVideo] = useState(false)
  const [metadata, setMetadata] = useState<StreamMetadata>(EMPTY_METADATA)
  const [frameSrc, setFrameSrc] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const lastFrameIdxRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number | null>(null)
  const inboundStreamRef = useRef<MediaStream | null>(null)
  const wsClientRef = useRef<WebSocketClient | null>(null)
  const rtcClientRef = useRef<WebRtcClient | null>(null)

  const apiBase = useMemo(() => baseUrl.replace(/\/$/, ''), [baseUrl])

  const updateFps = (frameIdx: number) => {
    const now = performance.now()
    if (lastFrameIdxRef.current !== null && lastFrameTimeRef.current !== null) {
      const frameDelta = frameIdx - lastFrameIdxRef.current
      const timeDelta = now - lastFrameTimeRef.current
      const currentFps = timeDelta > 0 ? (frameDelta * 1000) / timeDelta : 0
      setFps((prev) => Math.round(prev * 0.8 + currentFps * 0.2))
    }
    lastFrameIdxRef.current = frameIdx
    lastFrameTimeRef.current = now
  }

  const handleMetadata = (raw: unknown) => {
    try {
      setMetadata((prev) => {
        const parsed = parseMetadataMessage(raw, prev)
        updateFps(parsed.frameIdx)
        return parsed
      })
    } catch (err) {
      console.warn('[WebRTC] metadata handling failed', err)
      setMetadata({ ...EMPTY_METADATA })
    }
  }

  const attachStreamToVideo = (stream: MediaStream | null) => {
    const videoEl = videoRef.current
    if (!videoEl || !stream) return
    if (videoEl.srcObject !== stream) {
      videoEl.srcObject = stream
    }
    videoEl.muted = true
    videoEl.autoplay = true
    videoEl.playsInline = true
    videoEl.play().catch((err) => console.warn('[WebRTC] play failed', err))
  }

  const attachVideoElement = (el: HTMLVideoElement | null) => {
    videoRef.current = el
    if (el && inboundStreamRef.current) {
      attachStreamToVideo(inboundStreamRef.current)
      if (inboundStreamRef.current.getTracks().length > 0) {
        setHasVideo(true)
      }
    }
  }

  const attachTrack = (track: MediaStreamTrack) => {
    if (!inboundStreamRef.current) {
      inboundStreamRef.current = new MediaStream()
    }
    const stream = inboundStreamRef.current
    stream.addTrack(track)
    attachStreamToVideo(stream)
  }

  const connectWebRtc = async () => {
    setStreamState('connecting')
    setError(null)
    setMetadata({ ...EMPTY_METADATA })
    const client = new WebRtcClient({
      baseUrl: apiBase,
      onVideoTrack: (track) => {
        attachTrack(track)
        setHasVideo(true)
      },
      onMetadata: handleMetadata,
      onConnectionStateChange: (state) => {
        if (state === 'connected') setStreamState('playing')
        else if (state === 'failed') setStreamState('error')
        else if (state === 'disconnected' || state === 'closed') setStreamState('paused')
      },
    })
    rtcClientRef.current = client
    try {
      await client.connect()
    } catch (err) {
      console.warn('[WebRTC] connection failed', err)
      setStreamState('error')
      setError(err instanceof Error ? err.message : 'Failed to connect to streaming server')
      setHasVideo(false)
      setMetadata({ ...EMPTY_METADATA })
    }
  }

  const disconnect = () => {
    wsClientRef.current?.disconnect()
    wsClientRef.current = null
    rtcClientRef.current?.disconnect()
    rtcClientRef.current = null
    setStreamState('idle')
    setError(null)
    setHasVideo(false)
    setMetadata({ ...EMPTY_METADATA })
    setFrameSrc((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    if (videoRef.current?.srcObject instanceof MediaStream) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop())
      videoRef.current.srcObject = null
    }
    inboundStreamRef.current = null
  }

  const connect = async () => {
    disconnect()
    await connectWebRtc()
  }

  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase])

  useEffect(() => {
    if (streamState !== 'playing') {
      lastFrameIdxRef.current = null
      lastFrameTimeRef.current = null
      setFps(0)
    }
  }, [streamState])

  return {
    streamState,
    error,
    fps,
    hasVideo,
    metadata,
    frameSrc,
    videoRef,
    attachVideoElement,
    connect,
    disconnect,
  }
}
