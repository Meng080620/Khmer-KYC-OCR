import { Image } from 'expo-image';
import { router } from 'expo-router';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { FieldName } from '../api/types';
import { FIELD_ORDER } from '../api/types';
import { FieldRow } from '../components/FieldRow';
import { colors, radius, spacing } from '../theme';
import { fieldValue, useKyc } from '../store/kycStore';

function qualitySummary(result: NonNullable<ReturnType<typeof useKyc>['result']>) {
  const { quality } = result;
  const flags: string[] = [];
  if (quality.is_blurry) flags.push('blurry');
  if (quality.is_dark) flags.push('dark');
  if (quality.is_glared) flags.push('glare');
  return flags.length ? flags.join(', ') : 'good';
}

export default function ResultScreen() {
  const { result, edits, setEdit, reset } = useKyc();

  if (!result) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No result. Scan a document first.</Text>
        <Pressable style={styles.primaryButton} onPress={() => router.back()}>
          <Text style={styles.primaryButtonText}>Back to scanner</Text>
        </Pressable>
      </View>
    );
  }

  const faceUri = result.face?.image_base64
    ? `data:image/jpeg;base64,${result.face.image_base64}`
    : null;

  const onSubmit = () => {
    Alert.alert(
      'Review complete',
      'These values are OCR extraction only — they have not been verified against any authority. Submission to a KYC backend is a separate integration step.',
      [
        { text: 'Scan another', style: 'destructive', onPress: () => { reset(); router.back(); } },
        { text: 'Done', style: 'cancel' },
      ],
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {result.needs_review && (
        <View style={styles.reviewBanner}>
          <Text style={styles.reviewBannerText}>
            Some fields need review. Please check the highlighted values before submitting.
          </Text>
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Extracted fields</Text>
        {FIELD_ORDER.map((field) => (
          <FieldRow
            key={field}
            field={field}
            extracted={result.fields[field]}
            value={fieldValue(result, edits, field)}
            editable
            onChangeText={(value) => setEdit(field, value)}
          />
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Document quality</Text>
        <View style={styles.metaRow}>
          <Text style={styles.metaLabel}>Quality</Text>
          <Text style={styles.metaValue}>{qualitySummary(result)}</Text>
        </View>
        <View style={styles.metaRow}>
          <Text style={styles.metaLabel}>OCR engine</Text>
          <Text style={styles.metaValue}>{result.ocr_engine}</Text>
        </View>
        <View style={styles.metaRow}>
          <Text style={styles.metaLabel}>Detected face</Text>
          <Text style={styles.metaValue}>{result.face?.detected ? 'yes' : 'no'}</Text>
        </View>
      </View>

      {faceUri && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Detected portrait</Text>
          <Image source={{ uri: faceUri }} style={styles.face} contentFit="cover" />
        </View>
      )}

      <Text style={styles.disclaimer}>
        OCR extraction only. This result is not identity verification and does not confirm the
        document is authentic.
      </Text>

      <Pressable style={styles.primaryButton} onPress={onSubmit}>
        <Text style={styles.primaryButtonText}>Submit</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.md, gap: spacing.md },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.lg, padding: spacing.xl },
  emptyText: { color: colors.text, fontSize: 16 },
  reviewBanner: { backgroundColor: 'rgba(210,153,34,0.15)', borderRadius: radius.md, padding: spacing.md },
  reviewBannerText: { color: colors.warning, fontSize: 14, fontWeight: '500' },
  card: { backgroundColor: colors.surface, borderRadius: radius.md, overflow: 'hidden' },
  cardTitle: { color: colors.text, fontSize: 15, fontWeight: '700', padding: spacing.md, paddingBottom: spacing.xs },
  metaRow: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  metaLabel: { color: colors.textMuted, fontSize: 14 },
  metaValue: { color: colors.text, fontSize: 14, fontWeight: '500' },
  face: { height: 160, margin: spacing.md, marginTop: 0, borderRadius: radius.sm, backgroundColor: colors.surfaceAlt },
  disclaimer: { color: colors.textMuted, fontSize: 12, lineHeight: 18, textAlign: 'center', paddingHorizontal: spacing.md },
  primaryButton: { backgroundColor: colors.primary, paddingVertical: spacing.md, borderRadius: radius.md, alignItems: 'center' },
  primaryButtonText: { color: colors.primaryText, fontWeight: '700', fontSize: 16 },
});
