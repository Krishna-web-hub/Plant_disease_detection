// Mirrors the response shapes returned by deployment/api/app.py's /predict endpoint.

export type Severity = "none" | "moderate" | "moderate_to_high" | "high";

// A citation the user can click through and verify -- url is null for
// sources that aren't individually linkable (e.g. the internal treatment DB).
export interface SourceRef {
  source: string;
  url: string | null;
}

export interface Treatment {
  crop: string;
  disease: string;
  pathogen_type: string;
  pathogen: string | null;
  symptoms: string[];
  treatment: string[];
  prevention: string[];
  severity: Severity;
}

export interface DirectResult {
  type: "DIRECT";
  crop: string;
  category: string;
  disease: string;
  confidence: number;
  treatment: Treatment;
  rag_used: false;
}

export interface RagEnhancedResult {
  type: "RAG_ENHANCED";
  crop: string;
  category: string;
  model_confidence: number;
  treatment: Treatment;
  rag_reasoning: string;
  rag_sources: SourceRef[];
  rag_latency_seconds: number;
  rag_used: true;
}

export interface RagPendingResult {
  type: "RAG_PENDING";
  crop: string;
  category: string;
  model_confidence: number;
  treatment: Treatment;
  message: string;
  rag_used: false;
}

export interface UncertainResult {
  type: "UNCERTAIN";
  stage: "crop_classification" | "disease_classification";
  crop: string;
  category?: string;
  confidence: number;
  message: string;
  rag_used?: false;
}

export interface AiFallbackResult {
  type: "AI_FALLBACK";
  plant: string;
  disease: string;
  is_healthy: boolean;
  vision_confidence: number | null;
  reasoning: string;
  sources: SourceRef[];
  rag_latency_seconds: number;
  message: string;
}

export type PredictResponse =
  | DirectResult
  | RagEnhancedResult
  | RagPendingResult
  | UncertainResult
  | AiFallbackResult;

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  disease_crops_available: string[];
}
