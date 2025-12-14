<template>
  <div class="bg-white rounded-xl p-6 shadow-sm">
    <h2 class="text-lg font-semibold mb-4">History Summary</h2>

    <div v-if="summary" class="space-y-2 text-sm text-slate-700">
      <p>Total events: {{ summary.totalEvents }}</p>
      <p>Unique devices: {{ summary.devices }}</p>
      <p>Time range: {{ summary.timeRange }}</p>
    </div>

    <p v-else class="text-slate-400 text-sm">
      No data loaded yet.
    </p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  summary: any | null
}>()
</script>
