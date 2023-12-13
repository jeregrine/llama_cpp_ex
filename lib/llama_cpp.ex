defmodule LLamaCpp do
  use Rustler,
    otp_app: :llama_cpp,
    crate: :llamacpp

  def new(_llama_opts), do: :erlang.nif_error(:nif_not_loaded)
  def predict(_llama, _query), do: :erlang.nif_error(:nif_not_loaded)
  # def predict_stream(_llama, _query), do: :erlang.nif_error(:nif_not_loaded)
end
