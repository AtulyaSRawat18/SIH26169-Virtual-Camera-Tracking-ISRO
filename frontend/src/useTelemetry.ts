import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let socket: WebSocket | null = null
    let retry: number | undefined
    let stopped = false

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`)
      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => setTelemetry(JSON.parse(event.data) as Telemetry)
      socket.onclose = () => {
        setConnected(false)
        if (!stopped) retry = window.setTimeout(connect, 1000)
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      stopped = true
      if (retry !== undefined) window.clearTimeout(retry)
      socket?.close()
    }
  }, [])

  return { telemetry, connected }
}

