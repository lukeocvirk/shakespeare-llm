import ChatWindow from './components/ChatWindow'

export default function App() {
  return (
    <div>
      <header className="hero">
        <div className="container inner">
          <h4 className="subtitle">My work at IBM has been confidential, thus this website will instead be on a topic in Computer Science I find interesting. </h4>
          <h1 className="title">Large Language Models - Fall 2025 Work Term Report</h1>
          <h3 className="name">Luke Ocvirk</h3>
          <p className="subtitle">This website provides an overview of Large Language Models. First, I will provide a simple overview of the theory behind language models, and then you will be able to try it out yourself by
            prompting a mini demo language model I built; trained on the works of William Shakespeare.</p>
        </div>
      </header>

      <main className="container">
        <div className="grid" style={{ marginBottom: 16 }}>
          <section className="card">
            <h2 className="section-title">What are Large Language Models?</h2>
            <p>Large Language Models (LLMs) are a type of machine-learning model, trained on large amounts of text to predict the next text in a sequence.
              <br></br><br></br>Before training, the text is split into 'tokens', which are integer IDs that represent distinct characters, words, or other sets of characters.
              <br></br><br></br>The model learns vectorized embeddings for each token, and then uses multiple neural network layers (often, a transformer) to combine the embeddings with context from its prompt.
              The model then produces a probability distribution over all its tokens to determine the next token in the sequence of tokens.
              <br></br><br></br>By performing this operation repeatedly, the model outputs a series of tokens (which represent natural language), effectively 'completing' the prompt.
              <br></br><br></br>(Stryker, 2025)
            </p>
          </section>
          <section className="card">
            <h2 className="section-title">Tokenization</h2>
            <p>Large language models represent all text with tokens. Each token is represented by a unique integer ID. These tokens can represent any set of characters, and are created prior to training. A token can
              represent a single character or several characters, and often (but not always) correspond to words.
              <br></br><br></br>Since our model has to represent the statistical relationships between known words, if LLMs were limited to only representing one character at a time, their ability to learn how words
              relate to one another would be greatly limited due to a small set of tokens (one per character) with an immeasurely complex statistical probability space between them. With tokenization, tokens can
              represent words or subsets of words, allowing for a much more nuanced representation of inter-token relationships. This allows the model to much more efficiently and reliably learn the connections
              between words within natural language.
              <br></br><br></br>(Gondal, 2024)
              <br></br><br></br>The following is an example of how a sentence might be tokenized by a large language model:
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/kant_quote.png" style={{ width: '100%', maxWidth: '1000px' }} />
              </div>
              (Laforge, 2024)
            </p>
          </section>
          <section className="card">
            <h2 className="section-title">Embeddings</h2>
            <p>Once the set of tokens has been defined, the model needs to learn how the tokens relate to each other in context. To do this, tokens are converted into vectorized embeddings in high-dimensional space.
              <br></br><br></br>These embeddings provide a starting representation from which contextual relationships are learned in this high-dimensional space. Tokens that are more similar or related will often be
              close to each other in this vectorized space, and those that are less related will often be further apart.
              <br></br><br></br>By training a model on vast amounts of data (in the case of production-grade large language models), the model learns the embeddings and becomes capable of representing semantically
              valid natural language by having well-defined embeddings (along with model layers and good training objectives), despite not actually understanding the words that the embeddings represent.
              <br></br><br></br>(Gondal, 2024)
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/embeddings_visualization.jpg" style={{ width: '100%', maxWidth: '1000px' }} />
              </div>
              (Hassani, 2025)
            </p>
          </section>
          <section className="card">
            <h2 className="section-title">Autoregression (Next-Token Prediction)</h2>
            <p>Given a prompt, the model performs a series of next-token predictions.
              <br></br><br></br>Using its training, the model predicts the next token repeatedly by producing scores (logits), that are then converted into probabilities using a decoding strategy of choice
              (e.g., probablity-weighted sampling, greedy decoding, etc).
              <br></br><br></br>(Stryker, 2025)
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/llm_sampling.png" style={{ width: '100%', maxWidth: '1000px' }} />
              </div>
            </p>
          </section>
        </div>
        <ChatWindow />
      </main>

      <footer className="hero">
        <div className="container">
          <h2 className="section-title">Sources</h2>
          <ul className="sources-list">
            <li><a href="https://medium.com/thedeephub/all-you-need-to-know-about-tokenization-in-llms-7a801302cf54" target="_blank" rel="noopener noreferrer">All you need to know about Tokenization in LLMs (Gondal, 2024)</a></li>
            <li><a href="https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=bert_(bidirectional_encoder_representations_from_transformers)" target="_blank" rel="noopener noreferrer">Embeddings visualization (Hassani, 2025)</a></li>
            <li><a href="https://tokens-lpj6s2duga-ew.a.run.app/" target="_blank" rel="noopener noreferrer">Token visualizer (Laforge, 2024)</a></li>
            <li><a href="https://www.ibm.com/think/topics/large-language-models" target="_blank" rel="noopener noreferrer">What are LLMs? (Stryker, 2025)</a></li>
          </ul>
          <br></br>
          <p className="credit">Luke Ocvirk, 2025</p>
        </div>
      </footer>
    </div>
  )
}
