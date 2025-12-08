import type { MetadataMessage } from '../../domain/streaming/streamTypes'

export type WebSocketClientConfig = {
  url: string
  onOpen?: () => void
  onClose?: () => void
  onError?: (err: Event) => void
  onMetadata?: (data: MetadataMessage) => void
  onFrame?: (data: Blob | ArrayBuffer) => void
}

export class WebSocketClient {
  private ws: WebSocket | null = null
  private cfg: WebSocketClientConfig

  constructor(cfg: WebSocketClientConfig) {
    this.cfg = cfg
  }

  connect() {
    if (this.ws) return
    const ws = new WebSocket(this.cfg.url)
    ws.binaryType = 'blob'
    this.ws = ws

    ws.onopen = () => this.cfg.onOpen?.()

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const parsed = JSON.parse(event.data)
          this.cfg.onMetadata?.(parsed)
        } catch (err) {
          console.warn('[WS] JSON parse error', err)
        }
        return
      }
      this.cfg.onFrame?.(event.data as Blob | ArrayBuffer)
    }

    ws.onerror = (err) => this.cfg.onError?.(err)
    ws.onclose = () => {
      this.ws = null
      this.cfg.onClose?.()
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.onopen = null
      this.ws.onmessage = null
      this.ws.onclose = null
      this.ws.onerror = null
      this.ws.close()
      this.ws = null
    }
  }
}
