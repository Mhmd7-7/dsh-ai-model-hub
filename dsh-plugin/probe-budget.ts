/** Bound an individual discovery effect, including injected effects that do not honour their timeout. */
export async function withinBudget<T>(effect: () => Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve().then(effect),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`probe timed out after ${timeoutMs} ms`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
