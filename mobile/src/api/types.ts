/**
 * KYC data contract — mirrors the backend `KYCResult` / `RejectedImage` schemas exactly.
 * Keep in sync with `src/khmer_kyc_ocr/schemas.py`.
 */

export type FieldName =
  | 'document_type'
  | 'id_number'
  | 'name_kh'
  | 'name_en'
  | 'date_of_birth'
  | 'gender'
  | 'nationality'
  | 'place_of_birth'
  | 'address'
  | 'issue_date'
  | 'expiry_date';

export interface FieldResult {
  value: string | null;
  raw: string;
  confidence: number;
  valid: boolean;
}

export interface QualityReport {
  short_side_px: number;
  sharpness: number;
  is_blurry: boolean;
  brightness: number;
  is_dark: boolean;
  glare_ratio: number;
  is_glared: boolean;
  passed: boolean;
}

export interface FaceResult {
  detected: boolean;
  bbox: [number, number, number, number] | null;
  image_base64: string | null;
}

export interface KYCResult {
  fields: Partial<Record<FieldName, FieldResult>>;
  confidence: Partial<Record<FieldName, number>>;
  face: FaceResult | null;
  quality: QualityReport;
  ocr_engine: string;
  model_revision: string | null;
  schema_version: string;
  needs_review: boolean;
  review_reasons: string[];
}

export type RejectionReason =
  | 'undecodable'
  | 'too_small'
  | 'blurry'
  | 'glare'
  | 'dark'
  | 'card_not_found';

export interface RejectedImage {
  reason: RejectionReason;
  detail: string;
}

export const FIELD_LABELS: Record<FieldName, string> = {
  document_type: 'Document Type',
  id_number: 'ID Number',
  name_kh: 'Name (Khmer)',
  name_en: 'Name (English)',
  date_of_birth: 'Date of Birth',
  gender: 'Gender',
  nationality: 'Nationality',
  place_of_birth: 'Place of Birth',
  address: 'Address',
  issue_date: 'Issue Date',
  expiry_date: 'Expiry Date',
};

export const FIELD_ORDER: FieldName[] = [
  'document_type',
  'id_number',
  'name_kh',
  'name_en',
  'date_of_birth',
  'gender',
  'nationality',
  'place_of_birth',
  'address',
  'issue_date',
  'expiry_date',
];
