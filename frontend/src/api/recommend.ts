// src/api/recommend.ts

const BASE_URL = 'http://localhost:8000'

export async function recommendContextualFrequency(
    history: any[],
    currentContext: {
        event_time: string
        temperature: number
        humidity: number
        light_intensity: number
    }
) {
    const res = await fetch(`${BASE_URL}/recommend/contextual-frequency`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            history,
            context: currentContext,
        }),
    })

    if (!res.ok) {
        const text = await res.text()
        throw new Error(`Request failed: ${res.status} ${text}`)
    }
    return res.json()
}
