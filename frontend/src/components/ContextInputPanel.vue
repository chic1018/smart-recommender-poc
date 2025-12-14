<template>
  <div class="bg-white rounded-xl p-6 shadow-sm space-y-4">
    <h2 class="text-lg font-semibold">Current Context</h2>

    <!-- Time -->
    <div>
      <label class="block text-sm text-slate-600 mb-1">Time</label>
      <input
        type="datetime-local"
        v-model="local.event_time"
        class="w-full rounded-lg border px-3 py-2 text-sm"
      />
    </div>

    <!-- Temperature -->
    <div>
      <label class="block text-sm text-slate-600 mb-1">
        Temperature (°C)
      </label>
      <input
        type="number"
        step="0.1"
        v-model.number="local.temperature"
        class="w-full rounded-lg border px-3 py-2 text-sm"
      />
    </div>

    <!-- Humidity -->
    <div>
      <label class="block text-sm text-slate-600 mb-1">
        Humidity (%)
      </label>
      <input
        type="number"
        step="1"
        v-model.number="local.humidity"
        class="w-full rounded-lg border px-3 py-2 text-sm"
      />
    </div>

    <!-- Light -->
    <div>
      <label class="block text-sm text-slate-600 mb-1">
        Light Intensity
      </label>
      <input
        type="number"
        step="10"
        v-model.number="local.light_intensity"
        class="w-full rounded-lg border px-3 py-2 text-sm"
      />
    </div>

    <button
      class="mt-2 px-4 py-2 rounded-lg bg-slate-900 text-white text-sm hover:bg-slate-800"
      @click="emitUpdate"
    >
      Apply Context
    </button>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'

const props = defineProps<{
  modelValue: {
    event_time: string
    temperature: number
    humidity: number
    light_intensity: number
  }
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: typeof props.modelValue): void
}>()

const local = reactive({ ...props.modelValue })

watch(
  () => props.modelValue,
  v => Object.assign(local, v)
)

function emitUpdate() {
  emit('update:modelValue', { ...local })
}
</script>
