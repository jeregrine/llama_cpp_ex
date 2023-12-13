defmodule LlamaCppTest do
  use ExUnit.Case
  doctest LlamaCpp

  test "greets the world" do
    assert LlamaCpp.hello() == :world
  end
end
