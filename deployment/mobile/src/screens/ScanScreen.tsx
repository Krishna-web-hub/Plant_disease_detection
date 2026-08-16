import * as ImagePicker from "expo-image-picker";
import React, { useState } from "react";
import { ActivityIndicator, Button, Image, ScrollView, StyleSheet, Text, View } from "react-native";
import { ApiError, predict } from "../api/client";
import { ResultCard } from "../components/ResultCard";
import type { PredictResponse } from "../types";

type Status = "idle" | "loading" | "done" | "error";

export function ScanScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runPrediction(uri: string) {
    setImageUri(uri);
    setStatus("loading");
    setError(null);
    try {
      const response = await predict(uri);
      setResult(response);
      setStatus("done");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not reach the server.");
      setStatus("error");
    }
  }

  async function pickFromLibrary() {
    const { status: permission } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permission !== "granted") return;

    const picked = await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (!picked.canceled) {
      await runPrediction(picked.assets[0].uri);
    }
  }

  async function takePhoto() {
    const { status: permission } = await ImagePicker.requestCameraPermissionsAsync();
    if (permission !== "granted") return;

    const captured = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!captured.canceled) {
      await runPrediction(captured.assets[0].uri);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.heading}>Scan Leaf Photo</Text>

      <View style={styles.buttonRow}>
        <Button title="Take Photo" onPress={takePhoto} />
        <Button title="Choose from Library" onPress={pickFromLibrary} />
      </View>

      {imageUri && <Image source={{ uri: imageUri }} style={styles.preview} />}

      {status === "loading" && (
        <View style={styles.centered}>
          <ActivityIndicator size="large" />
          <Text>Analyzing...</Text>
        </View>
      )}

      {status === "error" && <Text style={styles.error}>{error}</Text>}

      {status === "done" && result && <ResultCard result={result} />}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 16 },
  heading: { fontSize: 24, fontWeight: "700", textAlign: "center" },
  buttonRow: { flexDirection: "row", justifyContent: "space-around" },
  preview: { width: "100%", height: 240, borderRadius: 12 },
  centered: { alignItems: "center", gap: 8 },
  error: { color: "#c0392b", textAlign: "center" },
});
