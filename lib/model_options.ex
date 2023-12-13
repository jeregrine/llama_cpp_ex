defmodule LlamaCpp.ModuleOptions do
  defstruct context_size: 512,
            seed: 0,
            f16_memory: true,
            m_lock: false,
            embeddings: false,
            low_vram: false,
            vocab_only: false,
            m_map: true,
            n_batch: 0,
            numa: false,
            n_gpu_layers: 0,
            main_gpu: "",
            tensor_split: ""
end
