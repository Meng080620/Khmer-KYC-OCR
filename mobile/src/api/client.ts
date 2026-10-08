import { API_BASE_URL } from '../config';
import { getApiToken } from './token';
import type { KYCResult, RejectedImage, RejectionReason } from './types';

export type SubmitError =
  | { kind: 'rejected'; reason: RejectionReason; detail: string }
  | { kind: 'http'; status: number; message: string }
  | { kind: 'network'; message: string };

export async function submitKyc(
  imageBase64: string,
  contentType: string = 'image/jpeg',
): Promise<KYCResult> {
  const token = await getApiToken();

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/v1/kyc`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Token': token,
      },
      body: JSON.stringify({ image_base64: imageBase64, content_type: contentType }),
    });
  } catch {
    throw { kind: 'network', message: 'Could not reach the OCR service.' } satisfies SubmitError;
  }

  if (response.status === 422) {
    const body = (await response.json()) as RejectedImage;
    throw { kind: 'rejected', reason: body.reason, detail: body.detail } satisfies SubmitError;
  }

  if (!response.ok) {
    throw {
      kind: 'http',
      status: response.status,
      message: `OCR service returned HTTP ${response.status}`,
    } satisfies SubmitError;
  }

  return (await response.json()) as KYCResult;
}
