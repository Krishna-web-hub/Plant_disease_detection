import type { PredictResponse, Treatment } from "../types";

function Pct({ value }: { value: number }) {
  return <span className="confidence">{Math.round(value * 100)}%</span>;
}

function TreatmentBlock({ treatment }: { treatment: Treatment }) {
  return (
    <div className="section">
      <h3>Treatment</h3>
      <ul>
        {treatment.treatment.map((step, i) => (
          <li key={i}>{step}</li>
        ))}
      </ul>
      <h3>Prevention</h3>
      <ul>
        {treatment.prevention.map((step, i) => (
          <li key={i}>{step}</li>
        ))}
      </ul>
    </div>
  );
}

export function ResultCard({ result }: { result: PredictResponse }) {
  if (result.type === "DIRECT") {
    return (
      <div className="card">
        <h2>{result.disease}</h2>
        <div className="row">
          <span className="label">Confidence</span>
          <Pct value={result.confidence} />
        </div>
        <TreatmentBlock treatment={result.treatment} />
      </div>
    );
  }

  if (result.type === "RAG_ENHANCED") {
    return (
      <div className="card">
        <h2>{result.treatment.disease} (uncertain — AI reviewed)</h2>
        <div className="row">
          <span className="label">Model confidence</span>
          <Pct value={result.model_confidence} />
        </div>
        <div className="section">
          <h3>Analysis</h3>
          <p>{result.rag_reasoning}</p>
        </div>
        <TreatmentBlock treatment={result.treatment} />
        <div className="section">
          <h3>Sources</h3>
          <ul>
            {result.rag_sources.map((source) => (
              <li key={source}>{source} ✓</li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  if (result.type === "RAG_PENDING") {
    return (
      <div className="card">
        <h2>{result.treatment.disease} (best guess)</h2>
        <p>{result.message}</p>
        <TreatmentBlock treatment={result.treatment} />
      </div>
    );
  }

  if (result.type === "AI_FALLBACK") {
    return (
      <div className="card">
        <span className="badge">AI diagnosis — outside trained crops</span>
        <h2>
          {result.plant} — {result.is_healthy ? "Healthy" : result.disease}
        </h2>
        {result.vision_confidence != null && (
          <div className="row">
            <span className="label">Vision model confidence</span>
            <Pct value={result.vision_confidence} />
          </div>
        )}
        <p>{result.message}</p>
        <div className="section">
          <h3>Analysis</h3>
          <p>{result.reasoning}</p>
        </div>
        {result.sources.length > 0 && (
          <div className="section">
            <h3>Sources</h3>
            <ul>
              {result.sources.map((source) => (
                <li key={source}>{source} ✓</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  // UNCERTAIN
  return (
    <div className="card">
      <h2>Uncertain</h2>
      <p>{result.message}</p>
    </div>
  );
}
