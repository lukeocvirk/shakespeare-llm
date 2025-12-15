import ChatWindow from './components/ChatWindow'

export default function App() {
  return (
    <div>
      <header className="hero">
        <div className="container inner">
          <h1 className="title">Shakespeare LLM — Demo</h1>
          <h3 className="name">Tiny Shakespeare Character-Level Model</h3>
          <p className="subtitle">Interact with a small character-level transformer trained on Tiny Shakespeare. Use the chat at the bottom to prompt the model.</p>
        </div>
      </header>

      <main className="container">
        <div className="grid" style={{ marginBottom: 16 }}>
          <section className="card">
            <h2 className="section-title">About</h2>
            <p>This demo exposes a small text-generation model trained on Tiny Shakespeare. It runs in Python on the backend; the frontend here sends prompts to the API and displays generated text.</p>
          </section>
          <section className="card">
            <h2 className="section-title">Usage</h2>
            <p>Type a line or a prompt in the chat and press Send. The frontend will POST to <code>/api/generate</code> by default — set <code>VITE_API_BASE</code> to change host/port.</p>
          </section>
        </div>

        {/* Chat window placed where diffusion demo was in the other project */}
        <ChatWindow />
      </main>

      <footer className="hero">
        <div className="container">
          <h2 className="section-title">Sources</h2>
          <p>Model and training code live in the repository root under <code>shakespeare-llm</code>. This frontend is a lightweight demo wrapper.</p>
        </div>
      </footer>
    </div>
  )
}
