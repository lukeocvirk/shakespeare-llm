import React, { useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type Message = { id: string; author: 'user' | 'bot'; text: string }

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    // welcome message
    setMessages([{ id: 'm0', author: 'bot', text: "Hello — try typing a prompt to the Shakespeare model (e.g. 'ROMEO:')" }])
  }, [])

  useEffect(() => {
    // auto-scroll to bottom
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const send = async () => {
    const text = input.trim()
    if (!text) return
    const userMsg: Message = { id: `u${Date.now()}`, author: 'user', text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const resp = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, max_new_tokens: 300 }),
      })
      if (!resp.ok) throw new Error('generate failed')
      const data = await resp.json()
      const botText = (data.generated_text ?? data.text ?? JSON.stringify(data)) as string
      const botMsg: Message = { id: `b${Date.now()}`, author: 'bot', text: botText }
      setMessages(prev => [...prev, botMsg])
    } catch (err: any) {
      const botMsg: Message = { id: `b${Date.now()}`, author: 'bot', text: `Error: ${err.message || err}` }
      setMessages(prev => [...prev, botMsg])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2 className="section-title">Shakespeare Chat</h2>
      <p>Chat with your trained Shakespeare LLM. The frontend will POST prompts to <code>/api/generate</code> on the backend.</p>

      <div ref={containerRef} style={{ maxHeight: 320, overflow: 'auto', padding: 8, border: '1px solid var(--muted)', borderRadius: 8, background: '#fff' }}>
        {messages.map(m => (
          <div key={m.id} style={{ margin: '8px 0', display: 'flex' }}>
            <div style={{ fontSize: 12, color: '#475569', width: 64 }}>{m.author === 'user' ? 'You' : 'Model'}</div>
            <div style={{ whiteSpace: 'pre-wrap', background: m.author === 'user' ? 'rgba(59,130,246,0.06)' : 'rgba(99,102,241,0.06)', padding: 8, borderRadius: 8, flex: 1 }}>{m.text}</div>
          </div>
        ))}
      </div>

      <div className="controls" style={{ marginTop: 12 }}>
        <input className="input" placeholder="Type a prompt (e.g. 'ROMEO:')" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') send() }} style={{ flex: 1 }} />
        <button className="btn" onClick={send} disabled={loading}>{loading ? 'Generating…' : 'Send'}</button>
      </div>
    </div>
  )
}
