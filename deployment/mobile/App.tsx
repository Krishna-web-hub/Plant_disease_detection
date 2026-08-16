import React from "react";
import { SafeAreaView, StyleSheet } from "react-native";
import { ScanScreen } from "./src/screens/ScanScreen";

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <ScanScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f2f2f2" },
});
