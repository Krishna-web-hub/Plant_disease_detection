import React from "react";
import { Linking, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { PredictResponse, Severity, SourceRef, Treatment } from "../types";

function Pct({ value }: { value: number }) {
  return <Text style={styles.confidence}>{Math.round(value * 100)}%</Text>;
}

const SEVERITY_LABELS: Record<Severity, string> = {
  none: "Healthy",
  moderate: "Moderate severity",
  moderate_to_high: "Moderate–high severity",
  high: "High severity",
};

const SEVERITY_COLORS: Record<Severity, { color: string; backgroundColor: string }> = {
  none: { color: "#1b5e20", backgroundColor: "#e3f6e5" },
  moderate: { color: "#8a5300", backgroundColor: "#fff3d6" },
  moderate_to_high: { color: "#9a4700", backgroundColor: "#ffe6cc" },
  high: { color: "#b71c1c", backgroundColor: "#fde0e0" },
};

function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <Text style={[styles.severity, SEVERITY_COLORS[severity]]}>{SEVERITY_LABELS[severity]}</Text>
  );
}

function SourceList({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Sources</Text>
      <Text style={styles.hint}>Verify these yourself — we don't ask you to just trust us.</Text>
      {sources.map((s) =>
        s.url ? (
          <TouchableOpacity key={s.url} onPress={() => Linking.openURL(s.url!)}>
            <Text style={styles.link}>• {s.source} ↗</Text>
          </TouchableOpacity>
        ) : (
          <Text key={s.source} style={styles.bullet}>
            • {s.source} ✓
          </Text>
        )
      )}
    </View>
  );
}

function TreatmentBlock({ treatment }: { treatment: Treatment }) {
  return (
    <View style={styles.section}>
      <SeverityBadge severity={treatment.severity} />
      <Text style={styles.sectionTitle}>Treatment</Text>
      {treatment.treatment.map((step, i) => (
        <Text key={i} style={styles.bullet}>
          • {step}
        </Text>
      ))}
      <Text style={styles.sectionTitle}>Prevention</Text>
      {treatment.prevention.map((step, i) => (
        <Text key={i} style={styles.bullet}>
          • {step}
        </Text>
      ))}
    </View>
  );
}

export function ResultCard({ result }: { result: PredictResponse }) {
  if (result.type === "DIRECT") {
    return (
      <View style={styles.card}>
        <Text style={styles.title}>{result.disease}</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Confidence</Text>
          <Pct value={result.confidence} />
        </View>
        <TreatmentBlock treatment={result.treatment} />
      </View>
    );
  }

  if (result.type === "RAG_ENHANCED") {
    return (
      <View style={styles.card}>
        <Text style={styles.title}>{result.treatment.disease} (uncertain — AI reviewed)</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Model confidence</Text>
          <Pct value={result.model_confidence} />
        </View>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analysis</Text>
          <Text style={styles.body}>{result.rag_reasoning}</Text>
        </View>
        <TreatmentBlock treatment={result.treatment} />
        <SourceList sources={result.rag_sources} />
      </View>
    );
  }

  if (result.type === "RAG_PENDING") {
    return (
      <View style={styles.card}>
        <Text style={styles.title}>{result.treatment.disease} (best guess)</Text>
        <Text style={styles.body}>{result.message}</Text>
        <TreatmentBlock treatment={result.treatment} />
      </View>
    );
  }

  if (result.type === "AI_FALLBACK") {
    return (
      <View style={styles.card}>
        <Text style={styles.badge}>AI DIAGNOSIS — OUTSIDE TRAINED CROPS</Text>
        <Text style={styles.title}>
          {result.plant} — {result.is_healthy ? "Healthy" : result.disease}
        </Text>
        {result.vision_confidence != null && (
          <View style={styles.row}>
            <Text style={styles.label}>Vision model confidence</Text>
            <Pct value={result.vision_confidence} />
          </View>
        )}
        <Text style={styles.body}>{result.message}</Text>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analysis</Text>
          <Text style={styles.body}>{result.reasoning}</Text>
        </View>
        <SourceList sources={result.sources} />
      </View>
    );
  }

  // UNCERTAIN
  return (
    <View style={styles.card}>
      <Text style={styles.title}>Uncertain</Text>
      <Text style={styles.body}>{result.message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { padding: 16, borderRadius: 12, backgroundColor: "#fff", gap: 8 },
  title: { fontSize: 20, fontWeight: "700" },
  row: { flexDirection: "row", justifyContent: "space-between" },
  label: { fontSize: 14, color: "#555" },
  confidence: { fontSize: 14, fontWeight: "600" },
  section: { marginTop: 8, gap: 2 },
  sectionTitle: { fontSize: 14, fontWeight: "700", marginBottom: 2 },
  bullet: { fontSize: 14, color: "#333" },
  body: { fontSize: 14, color: "#333" },
  hint: { fontSize: 12, color: "#777", marginBottom: 2 },
  link: { fontSize: 14, color: "#2e7d32", textDecorationLine: "underline" },
  badge: {
    alignSelf: "flex-start",
    fontSize: 11,
    fontWeight: "700",
    color: "#8a5300",
    backgroundColor: "#fff3d6",
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: 999,
    marginBottom: 4,
    overflow: "hidden",
  },
  severity: {
    alignSelf: "flex-start",
    fontSize: 12,
    fontWeight: "600",
    paddingVertical: 2,
    paddingHorizontal: 9,
    borderRadius: 999,
    marginBottom: 4,
    overflow: "hidden",
  },
});
