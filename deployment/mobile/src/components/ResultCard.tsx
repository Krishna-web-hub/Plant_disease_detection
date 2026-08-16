import React from "react";
import { StyleSheet, Text, View } from "react-native";
import type { PredictResponse, Treatment } from "../types";

function Pct({ value }: { value: number }) {
  return <Text style={styles.confidence}>{Math.round(value * 100)}%</Text>;
}

function TreatmentBlock({ treatment }: { treatment: Treatment }) {
  return (
    <View style={styles.section}>
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
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Sources</Text>
          {result.rag_sources.map((source) => (
            <Text key={source} style={styles.bullet}>
              • {source} ✓
            </Text>
          ))}
        </View>
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
        {result.sources.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Sources</Text>
            {result.sources.map((source) => (
              <Text key={source} style={styles.bullet}>
                • {source} ✓
              </Text>
            ))}
          </View>
        )}
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
});
