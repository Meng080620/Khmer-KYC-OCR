import { StyleSheet, Text, TextInput, View } from 'react-native';

import type { FieldName, FieldResult } from '../api/types';
import { FIELD_LABELS } from '../api/types';
import { colors, radius, spacing } from '../theme';
import { ConfidenceBadge, confidenceTier } from './ConfidenceBadge';

interface Props {
  field: FieldName;
  extracted: FieldResult | undefined;
  value: string;
  editable: boolean;
  onChangeText: (value: string) => void;
}

export function FieldRow({ field, extracted, value, editable, onChangeText }: Props) {
  const confidence = extracted?.confidence ?? 0;
  const invalid = extracted != null && !extracted.valid;
  const uncertain = extracted != null && confidenceTier(confidence) !== 'high';

  return (
    <View style={[styles.row, (invalid || uncertain) && styles.rowWarning]}>
      <Text style={styles.label}>{FIELD_LABELS[field]}</Text>

      <View style={styles.valueArea}>
        {editable ? (
          <TextInput
            style={styles.input}
            value={value}
            onChangeText={onChangeText}
            placeholder="Not detected"
            placeholderTextColor={colors.textMuted}
          />
        ) : (
          <Text style={[styles.value, !value && styles.missing]}>{value || '—'}</Text>
        )}
        {invalid && <Text style={styles.invalid}>needs review</Text>}
        {uncertain && !invalid && <Text style={styles.uncertain}>low confidence</Text>}
      </View>

      {extracted ? (
        <ConfidenceBadge confidence={confidence} />
      ) : (
        <Text style={styles.notDetected}>not detected</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  rowWarning: {
    backgroundColor: 'rgba(210, 153, 34, 0.08)',
  },
  label: {
    width: 130,
    color: colors.textMuted,
    fontSize: 13,
  },
  valueArea: {
    flex: 1,
  },
  value: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '500',
  },
  missing: {
    color: colors.textMuted,
  },
  input: {
    color: colors.text,
    fontSize: 16,
    padding: 0,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  invalid: {
    color: colors.danger,
    fontSize: 11,
    marginTop: 2,
  },
  uncertain: {
    color: colors.warning,
    fontSize: 11,
    marginTop: 2,
  },
  notDetected: {
    color: colors.textMuted,
    fontSize: 11,
    width: 60,
    textAlign: 'center',
  },
});
