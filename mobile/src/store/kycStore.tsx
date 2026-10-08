import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import type { FieldName, KYCResult } from '../api/types';

interface KycState {
  result: KYCResult | null;
  edits: Partial<Record<FieldName, string>>;
  setResult: (result: KYCResult | null) => void;
  setEdit: (field: FieldName, value: string) => void;
  reset: () => void;
}

const KycContext = createContext<KycState | null>(null);

export function KycProvider({ children }: { children: React.ReactNode }) {
  const [result, setResultState] = useState<KYCResult | null>(null);
  const [edits, setEdits] = useState<Partial<Record<FieldName, string>>>({});

  const setResult = useCallback((value: KYCResult | null) => {
    setResultState(value);
    setEdits({});
  }, []);

  const setEdit = useCallback((field: FieldName, value: string) => {
    setEdits((prev) => ({ ...prev, [field]: value }));
  }, []);

  const reset = useCallback(() => {
    setResultState(null);
    setEdits({});
  }, []);

  const value = useMemo(
    () => ({ result, edits, setResult, setEdit, reset }),
    [result, edits, setResult, setEdit, reset],
  );

  return <KycContext.Provider value={value}>{children}</KycContext.Provider>;
}

export function useKyc(): KycState {
  const ctx = useContext(KycContext);
  if (!ctx) {
    throw new Error('useKyc must be used within a KycProvider');
  }
  return ctx;
}

export function fieldValue(
  result: KYCResult,
  edits: Partial<Record<FieldName, string>>,
  field: FieldName,
): string {
  return edits[field] ?? result.fields[field]?.value ?? '';
}
