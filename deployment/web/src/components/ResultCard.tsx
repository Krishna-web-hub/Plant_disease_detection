import React, { useState } from "react";
import type { PredictResponse, Severity, SourceRef, TreatmentInfo, AgentAuditStep } from "../types";

function Pct({ value }: { value: number }) {
  return <span className="confidence">{Math.round(value * 100)}%</span>;
}

const SEVERITY_LABELS: Record<Severity, string> = {
  none: "Healthy",
  moderate: "Moderate severity",
  moderate_to_high: "Moderate–high severity",
  high: "High severity",
};

function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={`severity severity-${severity}`}>{SEVERITY_LABELS[severity]}</span>;
}

function SourceList({ sources }: { sources?: SourceRef[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="section">
      <h3>Sources & Citations</h3>
      <p className="hint">Grounded agricultural literature — independently verifiable.</p>
      <ul>
        {sources.map((s, idx) => (
          <li key={idx}>
            {s.url ? (
              <a href={s.url} target="_blank" rel="noopener noreferrer">
                {s.source} ↗
              </a>
            ) : (
              <>{s.source} ✓</>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function AuditTrail({ steps }: { steps?: AgentAuditStep[] }) {
  const [open, setOpen] = useState(true);
  if (!steps || steps.length === 0) return null;

  return (
    <div className="audit-trail">
      <div className="audit-header" onClick={() => setOpen(!open)} style={{ cursor: "pointer" }}>
        <span>🤖 Multi-Agent Verification Audit ({steps.length} Agents)</span>
        <span style={{ fontSize: "11px", color: "#666" }}>{open ? "▲ Hide" : "▼ Show"}</span>
      </div>
      {open && (
        <div className="audit-steps-container">
          {steps.map((s, idx) => {
            const statusClass = `status-${s.status.toLowerCase()}`;
            return (
              <div className="audit-step" key={idx}>
                <div className="audit-top">
                  <span className="audit-agent-name">
                    {s.agent === "SecuritySentinel" && "🛡️ Security Sentinel"}
                    {s.agent === "OODDetector" && "⚡ OOD & Energy Guard"}
                    {s.agent === "BotanicalCritic" && "🌿 Botanical Critic"}
                    {s.agent === "TaxonomyGuard" && "🧬 Taxonomy Guard"}
                    {s.agent === "SafetyRAGAgent" && "📚 Safety & Literature RAG"}
                    {!["SecuritySentinel", "OODDetector", "BotanicalCritic", "TaxonomyGuard", "SafetyRAGAgent"].includes(s.agent) && s.agent}
                  </span>
                  <span className={`audit-status-badge ${statusClass}`}>{s.status}</span>
                </div>
                <div className="audit-verdict-text">{s.verdict}</div>
                {s.original_hypothesis && (
                  <div style={{ fontSize: "11px", color: "#8a4000", marginTop: "4px" }}>
                    <strong>Overruled:</strong> {s.original_hypothesis} ➔ <strong>Corrected:</strong> {s.revised_hypothesis}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TreatmentSection({ treatment }: { treatment: TreatmentInfo }) {
  return (
    <div className="section">
      {treatment.severity && <SeverityBadge severity={treatment.severity} />}
      
      {treatment.organic && (
        <>
          <h3>Organic & Cultural Management</h3>
          <p>{treatment.organic}</p>
        </>
      )}

      {treatment.chemical && (
        <>
          <h3>Chemical Treatment Advisory</h3>
          <p>{treatment.chemical}</p>
        </>
      )}

      {treatment.treatment && treatment.treatment.length > 0 && (
        <>
          <h3>Treatment Steps</h3>
          <ul>
            {treatment.treatment.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </>
      )}

      {treatment.prevention && treatment.prevention.length > 0 && (
        <>
          <h3>Prevention</h3>
          <ul>
            {treatment.prevention.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </>
      )}

      {treatment.safety_notes && (
        <div className="warning-callout" style={{ background: "#fcf8e3", borderColor: "#f0ad4e" }}>
          <h4 style={{ color: "#8a6d3b" }}>⚠️ Phytotoxicity & Safety Advisory</h4>
          <p>{treatment.safety_notes}</p>
        </div>
      )}

      <SourceList sources={treatment.sources} />
    </div>
  );
}

export function ResultCard({ result }: { result: PredictResponse }) {
  // Case 1: OOD / Overfitting Intercepted
  if (result.type === "OOD_OVERFIT_INTERCEPTED") {
    const raw = result.intercepted_raw_prediction;
    return (
      <div className="card">
        <span className="badge badge-overfit">⚠️ Closed-Set Overfit Intercepted</span>
        <h2>{result.disease}</h2>
        <div className="row">
          <span className="label">Host Plant (Botanical)</span>
          <span style={{ fontWeight: 600 }}>{result.crop}</span>
        </div>
        <div className="row">
          <span className="label">Agent Consensus Confidence</span>
          <Pct value={result.confidence} />
        </div>

        <div className="warning-callout">
          <h4>Why Raw Model Was Overruled:</h4>
          <p>{raw.explanation}</p>
        </div>

        <div className="comparison-box">
          <div className="grid">
            <div className="col-raw">
              <strong>Raw Classifier (Closed-Set)</strong>
              <div>Crop: {raw.raw_crop} ({Math.round(raw.raw_crop_confidence * 100)}%)</div>
              <div>Disease: {raw.raw_disease} ({Math.round(raw.raw_disease_confidence * 100)}%)</div>
              <div>Free Energy: {raw.free_energy.toFixed(2)} (OOD Alert)</div>
            </div>
            <div className="col-corrected">
              <strong>Multi-Agent Resolution</strong>
              <div>Host: {result.crop}</div>
              <div>Condition: {result.disease}</div>
              <div>Causation: Insect Galls (Not Fungal)</div>
            </div>
          </div>
        </div>

        <AuditTrail steps={result.agent_audit_trail} />
        <TreatmentSection treatment={result.treatment} />
      </div>
    );
  }

  // Case 2: In-Domain Direct Verified
  if (result.type === "DIRECT_VERIFIED") {
    return (
      <div className="card">
        <span className="badge badge-verified">🌿 Verified In-Domain Diagnosis</span>
        <h2>{result.disease}</h2>
        <div className="row">
          <span className="label">Crop</span>
          <span style={{ fontWeight: 600 }}>{result.crop}</span>
        </div>
        <div className="row">
          <span className="label">Disease Classifier Confidence</span>
          <Pct value={result.confidence} />
        </div>
        <div className="row">
          <span className="label">Calibrated Confidence</span>
          <Pct value={result.calibrated_confidence} />
        </div>

        <AuditTrail steps={result.agent_audit_trail} />
        <TreatmentSection treatment={result.treatment} />
      </div>
    );
  }

  // Legacy DIRECT
  if (result.type === "DIRECT") {
    return (
      <div className="card">
        <h2>{result.disease}</h2>
        <div className="row">
          <span className="label">Host Crop</span>
          <span style={{ fontWeight: 600 }}>{result.crop}</span>
        </div>
        <div className="row">
          <span className="label">Confidence</span>
          <Pct value={result.confidence} />
        </div>
        <TreatmentSection treatment={result.treatment} />
      </div>
    );
  }

  // RAG_ENHANCED
  if (result.type === "RAG_ENHANCED") {
    return (
      <div className="card">
        <span className="badge">AI Literature Grounded</span>
        <h2>{result.treatment.disease || result.category}</h2>
        <div className="row">
          <span className="label">Host Crop</span>
          <span style={{ fontWeight: 600 }}>{result.crop}</span>
        </div>
        <div className="row">
          <span className="label">Model confidence</span>
          <Pct value={result.model_confidence} />
        </div>
        <div className="section">
          <h3>Literature Analysis</h3>
          <p>{result.rag_reasoning}</p>
        </div>
        <TreatmentSection treatment={result.treatment} />
      </div>
    );
  }

  // AI_FALLBACK
  if (result.type === "AI_FALLBACK") {
    return (
      <div className="card">
        <span className="badge">AI Multimodal Diagnosis</span>
        <h2>
          {result.plant} — {result.is_healthy ? "Healthy" : result.disease}
        </h2>
        {result.vision_confidence != null && (
          <div className="row">
            <span className="label">Vision confidence</span>
            <Pct value={result.vision_confidence} />
          </div>
        )}
        <p>{result.message}</p>
        <div className="section">
          <h3>Analysis</h3>
          <p>{result.reasoning}</p>
        </div>
        <SourceList sources={result.sources} />
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
