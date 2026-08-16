import { API_BASE_URL } from "../config";
import type { HealthResponse, PredictResponse } from "../types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new ApiError(response.status, "Health check failed");
  }
  return response.json();
}

export async function predict(imageUri: string, useRag = true): Promise<PredictResponse> {
  const formData = new FormData();
  // React Native's FormData accepts this shape for file uploads.
  formData.append("file", {
    uri: imageUri,
    name: "leaf.jpg",
    type: "image/jpeg",
  } as unknown as Blob);

  const response = await fetch(`${API_BASE_URL}/predict?use_rag=${useRag}`, {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(response.status, detail.detail ?? "Prediction failed");
  }

  return response.json();
}
