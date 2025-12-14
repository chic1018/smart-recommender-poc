<template>
  <div class="bg-white rounded-xl p-6 shadow-sm">
    <h2 class="text-lg font-semibold mb-4">Recommendations</h2>

    <div v-if="items.length === 0" class="text-slate-400 text-sm">
      Upload history to generate recommendations.
    </div>

    <ul v-else class="space-y-3">
      <li
          v-for="item in items"
          :key="item.device_id"
          class="flex justify-between items-center
               border rounded-lg px-4 py-3"
      >
        <span class="font-medium">{{ item.device_id }}</span>
        <span class="text-sm text-slate-500">
          score: {{ item.score.toFixed(4) }}
        </span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  items: { device_id: string; score: number }[]
}>()
</script>
