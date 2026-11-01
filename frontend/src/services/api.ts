import {
  UnifiedVerificationResponse,
  SystemHealthResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

/**
 * Fetch health status and backend capabilities
 */
export async function fetchHealth(): Promise<SystemHealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * Verify a textual claim
 */
export async function verifyTextClaim(claim: string): Promise<UnifiedVerificationResponse> {
  const res = await fetch(`${API_BASE}/verify/text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ claim: claim.trim() }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || `Text verification failed (${res.status})`);
  }

  return res.json();
}

/**
 * Verify an image file
 */
export async function verifyImageFile(file: File): Promise<UnifiedVerificationResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/verify/image`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || `Image verification failed (${res.status})`);
  }

  return res.json();
}

/**
 * Verify a video file with options
 */
export interface VideoVerifyOptions {
  enableClassifier?: boolean;
  verifyClaims?: boolean;
  whisperModel?: string;
  skipOcr?: boolean;
  maxClaims?: number;
  maxOcrFrames?: number;
  classifierFamily?: string;
}

export async function verifyVideoFile(
  file: File,
  options: VideoVerifyOptions = {}
): Promise<UnifiedVerificationResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('enable_classifier', String(options.enableClassifier ?? true));
  formData.append('verify_claims', String(options.verifyClaims ?? true));
  formData.append('whisper_model', options.whisperModel || 'base');
  formData.append('skip_ocr', String(options.skipOcr ?? false));
  formData.append('max_claims_to_verify', String(options.maxClaims ?? 1));
  formData.append('max_ocr_frames', String(options.maxOcrFrames ?? 15));
  formData.append('classifier_family', options.classifierFamily || 'videomae');

  const res = await fetch(`${API_BASE}/verify/video`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || `Video verification failed (${res.status})`);
  }

  return res.json();
}

/**
 * Verify an audio file with options
 */
export interface AudioVerifyOptions {
  verifyClaims?: boolean;
  whisperModel?: string;
  maxClaims?: number;
}

export async function verifyAudioFile(
  file: File,
  options: AudioVerifyOptions = {}
): Promise<UnifiedVerificationResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('verify_claims', String(options.verifyClaims ?? false));
  formData.append('whisper_model', options.whisperModel || 'base');
  formData.append('max_claims_to_verify', String(options.maxClaims ?? 1));

  const res = await fetch(`${API_BASE}/verify/audio`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || `Audio verification failed (${res.status})`);
  }

  return res.json();
}

// ─── SSE Streaming Types ───────────────────────────────────────────────────

export interface PipelineProgressEvent {
  stage: string;
  label: string;
  icon: string;
  detail: string;
  step: number;
  total_steps: number;
  raw?: string;
}

export interface PipelineStreamCallbacks {
  onStatus?: (evt: PipelineProgressEvent) => void;
  onProgress?: (evt: PipelineProgressEvent) => void;
  onResult?: (result: UnifiedVerificationResponse) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

/**
 * Stream text claim verification via SSE.
 *
 * Returns a cleanup function — call it to abort the stream early.
 *
 * Uses fetch + ReadableStream instead of EventSource so we can send
 * a GET request with a long query-string while staying inside the
 * same VITE_API_URL origin.
 */
export function streamTextVerification(
  claim: string,
  callbacks: PipelineStreamCallbacks
): () => void {
  const controller = new AbortController();
  const url = `${API_BASE}/verify/text/stream?claim=${encodeURIComponent(claim)}`;

  (async () => {
    try {
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'text/event-stream' },
      });

      if (!res.ok || !res.body) {
        const body = await res.text().catch(() => res.statusText);
        callbacks.onError?.(body || `Stream request failed (${res.status})`);
        callbacks.onDone?.();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const lines = part.split('\n');
          let eventType = 'message';
          let dataLine = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim();
            if (line.startsWith('data: ')) dataLine = line.slice(6).trim();
          }

          if (!dataLine) continue;

          try {
            const parsed = JSON.parse(dataLine);
            switch (eventType) {
              case 'status':
                callbacks.onStatus?.(parsed as PipelineProgressEvent);
                break;
              case 'progress':
                callbacks.onProgress?.(parsed as PipelineProgressEvent);
                break;
              case 'result':
                callbacks.onResult?.(parsed as UnifiedVerificationResponse);
                break;
              case 'error':
                callbacks.onError?.(parsed.message ?? 'Unknown pipeline error');
                break;
              case 'heartbeat':
                // ignore keep-alives
                break;
            }
          } catch {
            // Malformed SSE frame — ignore
          }
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        callbacks.onError?.(err?.message ?? 'Stream connection failed');
      }
    } finally {
      callbacks.onDone?.();
    }
  })();

  return () => controller.abort();
}

/**
 * Stream video verification via SSE (POST /verify/video/stream).
 *
 * Uploads the video file and streams real-time stage progress events.
 * Returns a cleanup function to abort the stream early.
 */
export function streamVideoVerification(
  file: File,
  options: VideoVerifyOptions,
  callbacks: PipelineStreamCallbacks
): () => void {
  const controller = new AbortController();
  const url = `${API_BASE}/verify/video/stream`;

  (async () => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('enable_classifier', String(options.enableClassifier ?? true));
      formData.append('verify_claims', String(options.verifyClaims ?? true));
      formData.append('whisper_model', options.whisperModel || 'base');
      formData.append('skip_ocr', String(options.skipOcr ?? false));
      formData.append('max_claims_to_verify', String(options.maxClaims ?? 1));
      formData.append('max_ocr_frames', String(options.maxOcrFrames ?? 15));
      formData.append('classifier_family', options.classifierFamily || 'videomae');

      const res = await fetch(url, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: { Accept: 'text/event-stream' },
      });

      if (!res.ok || !res.body) {
        const body = await res.text().catch(() => res.statusText);
        callbacks.onError?.(body || `Video stream request failed (${res.status})`);
        callbacks.onDone?.();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const lines = part.split('\n');
          let eventType = 'message';
          let dataLine = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim();
            if (line.startsWith('data: ')) dataLine = line.slice(6).trim();
          }

          if (!dataLine) continue;

          try {
            const parsed = JSON.parse(dataLine);
            switch (eventType) {
              case 'status':
                callbacks.onStatus?.(parsed as PipelineProgressEvent);
                break;
              case 'progress':
                callbacks.onProgress?.(parsed as PipelineProgressEvent);
                break;
              case 'result':
                callbacks.onResult?.(parsed as UnifiedVerificationResponse);
                break;
              case 'error':
                callbacks.onError?.(parsed.message ?? 'Unknown video pipeline error');
                break;
              case 'heartbeat':
                // ignore keep-alives
                break;
            }
          } catch {
            // Malformed SSE frame — ignore
          }
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        callbacks.onError?.(err?.message ?? 'Video stream connection failed');
      }
    } finally {
      callbacks.onDone?.();
    }
  })();

  return () => controller.abort();
}
