// Mirrors the response shapes returned by deployment/api/app.py's /predict endpoint.

export type Severity = "none" | "moderate" | "moderate_to_high" | "high";

export interface SourceRef {
  source: string;
  url: string | null;
}

export interface AgentAuditStep {
  agent: string;
  status: string;
  verdict: string;
  original_hypothesis?: string;
  revised_hypothesis?: string;
  metrics?: {
    free_energy?: number;
    max_logit?: number;
    raw_confidence?: number;
    calibrated_confidence?: number;
    overfit_flag?: boolean;
  };
}

export interface InterceptedRawPrediction {
  raw_crop: string;
  raw_crop_confidence: number;
  raw_disease: string;
  raw_disease_confidence: number;
  free_energy: number;
  explanation: string;
}

export interface TreatmentInfo {
  crop?: string;
  disease?: string;
  pathogen_type?: string;
  organic?: string;
  chemical?: string;
  safety_notes?: string;
  sources?: SourceRef[];
  treatment?: string[];
  prevention?: string[];
  severity?: Severity;
}

export interface OODOverfitResult {
  type: "OOD_OVERFIT_INTERCEPTED";
  message: string;
  crop: string;
  disease: string;
  is_in_domain: false;
  confidence: number;
  intercepted_raw_prediction: InterceptedRawPrediction;
  treatment: TreatmentInfo;
  agent_audit_trail: AgentAuditStep[];
}

export interface DirectVerifiedResult {
  type: "DIRECT_VERIFIED";
  crop: string;
  category: string;
  disease: string;
  confidence: number;
  calibrated_confidence: number;
  is_in_domain: true;
  treatment: TreatmentInfo;
  agent_audit_trail: AgentAuditStep[];
}

export interface DirectResult {
  type: "DIRECT";
  crop: string;
  category: string;
  disease: string;
  confidence: number;
  treatment: TreatmentInfo;
  rag_used: false;
}

export interface RagEnhancedResult {
  type: "RAG_ENHANCED";
  crop: string;
  category: string;
  model_confidence: number;
  treatment: TreatmentInfo;
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
  treatment: TreatmentInfo;
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
  | OODOverfitResult
  | DirectVerifiedResult
  | DirectResult
  | RagEnhancedResult
  | RagPendingResult
  | UncertainResult
  | AiFallbackResult;

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  orchestrator_ready?: boolean;
  disease_crops_available: string[];
}
