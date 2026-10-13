/** Bound an individual discovery effect, including injected effects that do not honour their timeout. */
export async function withinBudget(effect, timeoutMs) {
    let timer;
    try {
        return await Promise.race([
            Promise.resolve().then(effect),
            new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error(`probe timed out after ${timeoutMs} ms`)), timeoutMs);
            }),
        ]);
    }
    finally {
        if (timer !== undefined)
            clearTimeout(timer);
    }
}
