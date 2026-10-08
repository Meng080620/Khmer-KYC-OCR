import * as SecureStore from 'expo-secure-store';

import { KYC_TOKEN_KEY } from '../config';

export async function getApiToken(): Promise<string> {
  const stored = await SecureStore.getItemAsync(KYC_TOKEN_KEY);
  if (stored) {
    return stored;
  }
  return process.env.EXPO_PUBLIC_KYC_TOKEN ?? '';
}

export async function saveApiToken(token: string): Promise<void> {
  await SecureStore.setItemAsync(KYC_TOKEN_KEY, token, {
    keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
  });
}

export async function clearApiToken(): Promise<void> {
  await SecureStore.deleteItemAsync(KYC_TOKEN_KEY);
}
