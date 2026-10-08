/**
 * Runtime configuration. Secrets are never hardcoded here: the API token lives in
 * SecureStore (see src/api/token.ts), with an EXPO_PUBLIC_* env var as a dev fallback.
 *
 * On a physical device, 127.0.0.1 points at the phone, not your machine — set
 * EXPO_PUBLIC_KYC_API_URL to your machine's LAN address, e.g. http://192.168.1.10:8000.
 */
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_KYC_API_URL ?? 'http://127.0.0.1:8000';

export const KYC_TOKEN_KEY = 'khmer_kyc_ocr.api_token';
