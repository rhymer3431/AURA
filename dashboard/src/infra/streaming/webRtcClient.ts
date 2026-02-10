import type { MetadataMessage } from '../../domain/streaming/streamTypes'

export type WebRtcClientConfig = {
  baseUrl: string
  onVideoTrack?: (track: MediaStreamTrack) => void
  onMetadata?: (data: MetadataMessage) => void
  onConnectionStateChange?: (state: RTCPeerConnectionState) => void
}

export class WebRtcClient {
  private pc: RTCPeerConnection | null = null
  private metadataChannel: RTCDataChannel | null = null
  private readonly cfg: WebRtcClientConfig

  constructor(cfg: WebRtcClientConfig) {
    this.cfg = cfg
  }

  async connect(): Promise<void> {
    if (this.pc) return
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    })
    this.pc = pc

    pc.onconnectionstatechange = () => {
      this.cfg.onConnectionStateChange?.(pc.connectionState)
    }

    pc.ontrack = (event) => {
      if (event.track.kind === 'video') {
        this.cfg.onVideoTrack?.(event.track)
      }
    }

    this.metadataChannel = pc.createDataChannel('metadata')
    this.bindMetadataChannel(this.metadataChannel)

    pc.ondatachannel = (event) => {
      if (event.channel.label === 'metadata') {
        console.log(event.channel)
        this.bindMetadataChannel(event.channel)
      }
    }

    pc.addTransceiver('video', { direction: 'recvonly' })

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    const localCandidates = await this.gatherLocalCandidates(pc)

    const payload = {
      sdp: pc.localDescription?.sdp,
      type: pc.localDescription?.type,
      iceCandidates: localCandidates,
    }

    const res = await fetch(`${this.cfg.baseUrl}/webrtc/offer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      throw new Error(`Offer failed: ${res.status}`)
    }

    const answer = await res.json()
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type })

    const remoteCandidates = answer.iceCandidates ?? answer.ice_candidates ?? []
    for (const c of remoteCandidates) {
      if (!c || !c.candidate) continue
      await pc.addIceCandidate({
        candidate: c.candidate,
        sdpMid: c.sdpMid ?? undefined,
        sdpMLineIndex: c.sdpMLineIndex ?? undefined,
      })
    }
  }

  disconnect() {
    this.metadataChannel?.close()
    this.metadataChannel = null
    if (this.pc) {
      this.pc.ontrack = null
      this.pc.onconnectionstatechange = null
      this.pc.close()
      this.pc = null
    }
  }

  private bindMetadataChannel(channel: RTCDataChannel) {
    channel.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data)

        this.cfg.onMetadata?.(parsed)
      } catch (err) {
        console.warn('[WebRTC] metadata parse error', err)
      }
    }
    channel.onclose = () => {
      this.cfg.onConnectionStateChange?.('disconnected')
    }
    this.metadataChannel = channel
  }

  private gatherLocalCandidates(pc: RTCPeerConnection) {
    return new Promise<any[]>((resolve) => {
      const candidates: any[] = []
      const cleanup = () => {
        pc.removeEventListener('icecandidate', onCandidate)
        pc.removeEventListener('icegatheringstatechange', onState)
      }
      const onCandidate = (e: RTCPeerConnectionIceEvent) => {
        if (e.candidate) {
          candidates.push({
            candidate: e.candidate.candidate,
            sdpMid: e.candidate.sdpMid,
            sdpMLineIndex: e.candidate.sdpMLineIndex,
          })
        }
        if (!e.candidate && pc.iceGatheringState === 'complete') {
          cleanup()
          resolve(candidates)
        }
      }
      const onState = () => {
        if (pc.iceGatheringState === 'complete') {
          cleanup()
          resolve(candidates)
        }
      }
      pc.addEventListener('icecandidate', onCandidate)
      pc.addEventListener('icegatheringstatechange', onState)
      if (pc.iceGatheringState === 'complete') {
        cleanup()
        resolve(candidates)
      }
    })
  }
}
