import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '../theme';

export function confidenceTier(confidence: number): 'high' | 'medium' | 'low' {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}

const TIER_COLOR: Record<'high' | 'medium' | 'low', string> = {
  high: colors.success,
  medium: colors.warning,
  low: colors.danger,
};

export function ConfidenceBadge({ confidence }: { confidence: number }) {
  const tier = confidenceTier(confidence);
  return (
    <View style={[styles.badge, { borderColor: TIER_COLOR[tier] }]}>
      <Text style={[styles.text, { color: TIER_COLOR[tier] }]}>
        {Math.round(confidence * 100)}%
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderWidth: 1,
    borderRadius: radius.pill,
    paddingHorizontal: 8,
    paddingVertical: 2,
    minWidth: 44,
    alignItems: 'center',
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
  },
});
