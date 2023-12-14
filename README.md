# LlamaCpp

**TODO: Add description**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `llama_cpp` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llama_cpp, "~> 0.1.0"}
  ]
end
```

## Usage

```elixir
 {:ok, llama} = LLamaCpp.new("./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf")
 LLamaCpp.predict(llama, "hello my name is")
 LLamaCpp.embeddings(llama, "hello my name is")
 LLamaCpp.free_model(llama)
```
