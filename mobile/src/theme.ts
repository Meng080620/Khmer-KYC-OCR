/** Minimal design tokens so the app has a consistent, deliberate look. */

export const colors = {
  background: '#0E1116',
  surface: '#161B22',
  surfaceAlt: '#1F2733',
  border: '#2A3442',
  text: '#E6EDF3',
  textMuted: '#8B98A9',
  primary: '#2F81F7',
  primaryText: '#FFFFFF',
  danger: '#F85149',
  warning: '#D29922',
  success: '#3FB950',
  overlay: 'rgba(0, 0, 0, 0.55)',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 20,
  pill: 999,
} as const;
