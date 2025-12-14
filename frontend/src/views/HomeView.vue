<template>
  <div class="min-h-screen bg-slate-50 px-6 py-10">
    <div class="max-w-6xl mx-auto space-y-10">
      <!-- Header -->
      <header class="space-y-2">
        <h1 class="text-3xl font-semibold text-slate-900">
          Smart Furniture Recommender (POC)
        </h1>
        <p class="text-slate-600">
          Upload user interaction history and preview device recommendations.
        </p>
      </header>

      <!-- Main -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <FileDropZone @loaded="onHistoryLoaded"/>
        <HistoryPreview :summary="historySummary"/>
        <ContextInputPanel v-model="currentContext" @update:modelValue="updateRecommendations"/>
      </div>


      <!-- Result -->
      <RecommendationPanel :items="recommendations"/>
    </div>
  </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'

const loading = ref(false)
const recommendations = ref<any[]>([])
const historySummary = ref<{
  totalEvents: number
  devices: number
  timeRange: string
} | null>(null)

import FileDropZone from '@/components/FileDropZone.vue'
import HistoryPreview from '@/components/HistoryPreview.vue'
import RecommendationPanel from '@/components/RecommendationPanel.vue'
import {recommendContextualFrequency} from '@/api/recommend'
import ContextInputPanel from '@/components/ContextInputPanel.vue'


const currentContext = ref({
  event_time: new Date().toISOString().slice(0, 16),
  temperature: 22,
  humidity: 45,
  light_intensity: 300,
})

let history = ref<any>()

async function updateRecommendations() {
  if (!history.value.length) return

  try {
    loading.value = true

    const result = await recommendContextualFrequency(history.value, {
      ...currentContext.value,
      event_time: new Date(currentContext.value.event_time).toISOString(),
    })

    console.log(result)

    recommendations.value = result.recommendations ?? []
  } catch (err) {
    console.error('[HomeView] recommendation failed:', err)
    recommendations.value = []
  } finally {
    loading.value = false
  }
}


async function onHistoryLoaded(data: any[]) {
  console.log('[HomeView] received history:', data)

  try {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('History is empty or invalid')
    }

    historySummary.value = {
      totalEvents: data.length,
      devices: [...new Set(data.map(d => d.device_id))].length,
      timeRange: `${data[0].event_time} ~ ${data[data.length - 1].event_time}`,
    }

    history = ref(data)

    await updateRecommendations()
  } catch (err) {
    console.error('[HomeView] recommendation failed:', err)
  }
}


</script>
