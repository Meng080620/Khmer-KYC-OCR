import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Switch, Text, View } from 'react-native';

import { submitKyc, type SubmitError } from '../api/client';
import { colors, radius, spacing } from '../theme';
import { useKyc } from '../store/kycStore';

const REJECTION_HINTS: Record<string, string> = {
  blurry: 'Photo is blurry — hold the phone steady and retake.',
  glare: 'Too much glare — move out of direct light.',
  dark: 'Too dark — add more light on the card.',
  too_small: 'Card too small in frame — move closer.',
  card_not_found: 'No card found — keep the whole card inside the frame.',
  undecodable: 'Could not read the image — use a JPEG or PNG.',
};

export default function ScannerScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const { setResult } = useKyc();

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<SubmitError | null>(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [ready, setReady] = useState(false);

  const processImage = useCallback(
    async (base64: string) => {
      setBusy(true);
      setError(null);
      try {
        const result = await submitKyc(base64);
        setResult(result);
        router.push('/result');
      } catch (err) {
        setError(err as SubmitError);
      } finally {
        setBusy(false);
      }
    },
    [setResult],
  );

  const capture = useCallback(async () => {
    const photo = await cameraRef.current?.takePictureAsync({ base64: true, quality: 0.9 });
    if (photo?.base64) {
      await processImage(photo.base64);
    }
  }, [processImage]);

  const pickImage = useCallback(async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError({ kind: 'http', status: 0, message: 'Photo library permission is required.' });
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      base64: true,
      quality: 0.9,
    });
    if (!result.canceled && result.assets[0]?.base64) {
      await processImage(result.assets[0].base64);
    }
  }, [processImage]);

  useEffect(() => {
    if (!autoCapture || !ready || busy) return;
    const timer = setTimeout(() => {
      void capture();
    }, 1800);
    return () => clearTimeout(timer);
  }, [autoCapture, ready, busy, capture]);

  if (!permission) {
    return <View style={styles.container} />;
  }
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>Camera access is required to scan documents.</Text>
        <Pressable style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Grant camera permission</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="back"
        onCameraReady={() => setReady(true)}
      />

      <View style={styles.guideOverlay} pointerEvents="none">
        <View style={styles.guideFrame}>
          <View style={[styles.corner, styles.cornerTL]} />
          <View style={[styles.corner, styles.cornerTR]} />
          <View style={[styles.corner, styles.cornerBL]} />
          <View style={[styles.corner, styles.cornerBR]} />
        </View>
        <Text style={styles.guideHint}>
          Line up the card inside the frame on a plain background.
        </Text>
      </View>

      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>
            {error.kind === 'rejected' ? REJECTION_HINTS[error.reason] ?? error.detail : error.message}
          </Text>
        </View>
      )}

      {busy && (
        <View style={styles.busyOverlay}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.busyText}>Reading document…</Text>
        </View>
      )}

      <View style={styles.controls}>
        <View style={styles.autoRow}>
          <Text style={styles.autoLabel}>Auto-capture</Text>
          <Switch value={autoCapture} onValueChange={setAutoCapture} trackColor={{ true: colors.primary }} />
        </View>

        <View style={styles.buttonRow}>
          <Pressable style={styles.secondaryButton} onPress={pickImage} disabled={busy}>
            <Text style={styles.secondaryButtonText}>Upload</Text>
          </Pressable>
          <Pressable style={styles.captureButton} onPress={capture} disabled={busy}>
            <View style={styles.captureInner} />
          </Pressable>
          <View style={styles.spacer} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  permissionText: { color: colors.text, textAlign: 'center', marginBottom: spacing.lg, paddingHorizontal: spacing.xl },
  primaryButton: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: radius.md,
    alignSelf: 'center',
  },
  primaryButtonText: { color: colors.primaryText, fontWeight: '600', fontSize: 16 },
  guideOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', padding: spacing.lg },
  guideFrame: { width: '88%', aspectRatio: 1.586, justifyContent: 'center' },
  corner: { position: 'absolute', width: 28, height: 28, borderColor: colors.primary },
  cornerTL: { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 6 },
  cornerTR: { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 6 },
  cornerBL: { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 6 },
  cornerBR: { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 6 },
  guideHint: {
    color: colors.text,
    textAlign: 'center',
    marginTop: spacing.md,
    backgroundColor: colors.overlay,
    borderRadius: radius.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    overflow: 'hidden',
  },
  errorBanner: { position: 'absolute', top: spacing.md, left: spacing.md, right: spacing.md, backgroundColor: colors.danger, borderRadius: radius.md, padding: spacing.md },
  errorText: { color: '#fff', fontSize: 14, fontWeight: '500' },
  busyOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: colors.overlay, alignItems: 'center', justifyContent: 'center' },
  busyText: { color: colors.text, marginTop: spacing.md, fontSize: 16 },
  controls: { position: 'absolute', bottom: 0, left: 0, right: 0, padding: spacing.lg, backgroundColor: 'rgba(0,0,0,0.35)' },
  autoRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, marginBottom: spacing.md },
  autoLabel: { color: colors.text, fontSize: 14 },
  buttonRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  secondaryButton: { paddingVertical: spacing.md, paddingHorizontal: spacing.lg, borderRadius: radius.md, backgroundColor: colors.surfaceAlt },
  secondaryButtonText: { color: colors.text, fontWeight: '600' },
  captureButton: { width: 72, height: 72, borderRadius: 36, borderWidth: 4, borderColor: '#fff', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.15)' },
  captureInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#fff' },
  spacer: { width: 72 },
});
