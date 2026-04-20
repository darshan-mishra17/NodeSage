import { Ollama } from "ollama";
import { loadConfig } from "./config.js";

const ollama = new Ollama();
const BATCH_SIZE = 50;
const CONCURRENCY = 3;
const FALLBACK_CHAR_LIMITS = [
  24000,
  16000,
  12000,
  8000,
  6000,
  4000,
  2500,
  1600,
  1000,
  700,
  500,
  350,
  250,
  160,
  100,
];

function isContextLengthError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return /input length exceeds the context length/i.test(error.message);
}

function truncateForEmbedding(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;

  // Preserve both header/prefix and tail context when shrinking very long chunks.
  const headChars = Math.floor(maxChars * 0.75);
  const tailChars = maxChars - headChars;
  const head = text.slice(0, headChars);
  const tail = text.slice(-tailChars);

  return `${head}\n\n[...truncated for embedding...]\n\n${tail}`;
}

async function embedSingleWithFallback(model: string, text: string): Promise<number[]> {
  let lastError: unknown;

  for (const limit of FALLBACK_CHAR_LIMITS) {
    try {
      const input = truncateForEmbedding(text, limit);
      const response = await ollama.embed({ model, input });
      return response.embeddings[0];
    } catch (error) {
      lastError = error;
      if (!isContextLengthError(error)) {
        throw error;
      }
    }
  }

  // Absolute last resort for tiny context embedding models: embed a tiny marker.
  try {
    const marker = text.slice(0, 80).replace(/\s+/g, " ").trim();
    const response = await ollama.embed({
      model,
      input: marker ? `[truncated] ${marker}` : "[truncated]",
    });
    return response.embeddings[0];
  } catch (error) {
    lastError = error;
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Failed to embed text after fallback attempts.");
}

export async function embedText(text: string): Promise<number[]> {
  const config = await loadConfig();
  return embedSingleWithFallback(config.embedModel, text);
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  const config = await loadConfig();
  const allEmbeddings: number[][] = new Array(texts.length);

  // Split into batches
  const batches: { texts: string[]; startIdx: number }[] = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    batches.push({
      texts: texts.slice(i, i + BATCH_SIZE),
      startIdx: i,
    });
  }

  // Process batches with concurrency
  for (let i = 0; i < batches.length; i += CONCURRENCY) {
    const concurrentBatches = batches.slice(i, i + CONCURRENCY);
    const results = await Promise.all(
      concurrentBatches.map(async (batch) => {
        try {
          const response = await ollama.embed({
            model: config.embedModel,
            input: batch.texts,
          });
          return { embeddings: response.embeddings, startIdx: batch.startIdx };
        } catch (error) {
          if (!isContextLengthError(error)) {
            throw error;
          }

          const embeddings = await Promise.all(
            batch.texts.map((text) => embedSingleWithFallback(config.embedModel, text))
          );
          return { embeddings, startIdx: batch.startIdx };
        }
      })
    );

    for (const result of results) {
      for (let j = 0; j < result.embeddings.length; j++) {
        allEmbeddings[result.startIdx + j] = result.embeddings[j];
      }
    }
  }

  return allEmbeddings;
}
