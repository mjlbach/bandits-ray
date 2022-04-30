FullyConnectedNetwork(
  (_hidden_layers): Sequential(
    (0): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU()
      )
    )
    (1): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): ReLU()
      )
    )
    (2): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): ReLU()
      )
    )
  )
  (_value_branch_separate): Sequential(
    (0): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU()
      )
    )
    (1): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): ReLU()
      )
    )
    (2): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): ReLU()
      )
    )
  )
  (_value_branch): SlimFC(
    (_model): Sequential(
      (0): Linear(in_features=128, out_features=1, bias=True)
    )
  )
)

ComplexInputNetwork(
  (post_fc_stack): FullyConnectedNetwork(
    (_hidden_layers): Sequential(
      (0): SlimFC(
        (_model): Sequential(
          (0): Linear(in_features=256, out_features=128, bias=True)
          (1): ReLU()
        )
      )
      (1): SlimFC(
        (_model): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): ReLU()
        )
      )
    )
    (_value_branch_separate): Sequential(
      (0): SlimFC(
        (_model): Sequential(
          (0): Linear(in_features=256, out_features=128, bias=True)
          (1): ReLU()
        )
      )
      (1): SlimFC(
        (_model): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): ReLU()
        )
      )
    )
    (_value_branch): SlimFC(
      (_model): Sequential(
        (0): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
  (logits_layer): SlimFC(
    (_model): Sequential(
      (0): Linear(in_features=128, out_features=2, bias=True)
    )
  )
  (value_layer): SlimFC(
    (_model): Sequential(
      (0): Linear(in_features=256, out_features=1, bias=True)
    )
  )
)
