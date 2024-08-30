# Model Construction Utils (moconut)

```Python
    import moconut

    model = moconut([
        "in -> (x0:t<:,28,28>)",
        "(x0:t<:,28,28>) -> flatten -> (x0:t<:,784>)",
        "(x0:t<:,784>) -> linear<784,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> out"
    ])
```