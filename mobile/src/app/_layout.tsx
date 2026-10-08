import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { colors } from '../theme';
import { KycProvider } from '../store/kycStore';

export default function RootLayout() {
  return (
    <KycProvider>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.surface },
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: '600' },
          contentStyle: { backgroundColor: colors.background },
        }}
      >
        <Stack.Screen name="index" options={{ title: 'Scan ID Card' }} />
        <Stack.Screen name="result" options={{ title: 'Review KYC' }} />
      </Stack>
    </KycProvider>
  );
}
