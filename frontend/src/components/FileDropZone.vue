<template>
  <div
    v-bind="getRootProps()"
    class="border-2 border-dashed rounded-xl p-10 cursor-pointer transition
           bg-white hover:bg-slate-50"
    :class="isDragActive ? 'border-blue-500' : 'border-slate-300'"
  >
    <input v-bind="getInputProps()" />

    <div class="text-center space-y-3">
      <p class="text-lg font-medium text-slate-700">
        Drop history file here
      </p>
      <p class="text-sm text-slate-500">
        or click to select (CSV / JSON)
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useDropzone } from 'vue3-dropzone'

const emit = defineEmits<{
  (e: 'loaded', data: any[]): void
}>()

const { getRootProps, getInputProps, isDragActive } = useDropzone({
  multiple: false,
  onDrop: async (files) => {
    const file = files[0]
    const text = await file.text()

    let data: any[] = []

    if (file.name.endsWith('.json')) {
      data = JSON.parse(text)
    } else {
      const lines = text
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0)

      const header = lines[0]
      const rows = lines.slice(1)
      const keys = header.split(',')

      data = rows.map(row => {
        const values = row.split(',')

        return {
          event_time: values[keys.indexOf('event_time')],
          device_id: values[keys.indexOf('device_id')],
          action: values[keys.indexOf('action')],
          temperature: Number(values[keys.indexOf('temperature')]),
          humidity: Number(values[keys.indexOf('humidity')]),
          light_intensity: Number(values[keys.indexOf('light_intensity')]),
        }
      })
    }

    console.log('[DropZone] parsed history:', data)

    emit('loaded', data)
  }
})
</script>
