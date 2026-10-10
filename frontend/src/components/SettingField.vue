<script setup>
import { computed } from "vue";

const props = defineProps({ spec: Object, modelValue: null, options: { type: Array, default: () => [] } });
const emit = defineEmits(["update:modelValue", "clear"]);
const id = computed(() => "set-" + props.spec.key.replace(/\./g, "-"));
const v = computed({ get: () => props.modelValue, set: (x) => emit("update:modelValue", x) });
</script>

<template>
  <label v-if="spec.type === 'bool'" class="check">
    <input :id="id" v-model="v" type="checkbox" :disabled="spec.locked" />
    <span>{{ spec.label }}
      <small v-if="spec.help">{{ spec.help }}</small>
      <small v-if="spec.locked" class="lock">Set by {{ spec.env }}</small>
    </span>
  </label>
  <label v-else class="field">
    <span>{{ spec.label }} <small v-if="spec.locked" class="lock">· set by {{ spec.env }}</small></span>
    <div class="row" style="flex-wrap:nowrap">
      <input v-if="spec.secret" :id="id" v-model="v" type="password" autocomplete="off" :disabled="spec.locked"
             :placeholder="spec.is_set ? '•••••••• saved — leave blank to keep' : 'Not set'" />
      <input v-else-if="spec.type === 'int' || spec.type === 'float'" :id="id" v-model.number="v" type="number"
             :step="spec.type === 'float' ? 0.05 : 1" :disabled="spec.locked" />
      <template v-else>
        <input :id="id" v-model="v" type="text" :disabled="spec.locked" :list="options.length ? id + '-list' : undefined" />
        <datalist v-if="options.length" :id="id + '-list'"><option v-for="o in options" :key="o" :value="o" /></datalist>
      </template>
      <button v-if="spec.secret && spec.is_set && !spec.locked" type="button" class="btn small" @click="emit('clear')">Clear</button>
    </div>
    <small v-if="spec.help">{{ spec.help }}</small>
  </label>
</template>

<style scoped>
.lock { color: var(--warn); }
</style>
