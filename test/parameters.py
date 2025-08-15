from linmult import LinMulT, load_config


config_40 = load_config("configs/MTL/stage2/model_40.yaml")
model_40 = LinMulT(config_40)

config_100 = load_config("configs/MTL/stage2/model_100.yaml")
model_100 = LinMulT(config_100)

trainable_params_40 = sum(p.numel() for p in model_40.parameters() if p.requires_grad)
print(f"Trainable parameters (40): {trainable_params_40}")

trainable_params_100 = sum(p.numel() for p in model_100.parameters() if p.requires_grad)
print(f"Trainable parameters (100): {trainable_params_100}")

del model_40.output_heads['tmm_wavlm_baseplus']
del model_40.output_heads['tmm_clip']
del model_40.output_heads['tmm_xml_roberta']

trainable_params_40 = sum(p.numel() for p in model_40.parameters() if p.requires_grad)
print(f"Trainable parameters (40) without tmm heads: {trainable_params_40}")

del model_100.output_heads['tmm_wavlm_baseplus']
del model_100.output_heads['tmm_clip']
del model_100.output_heads['tmm_xml_roberta']

trainable_params_100 = sum(p.numel() for p in model_100.parameters() if p.requires_grad)
print(f"Trainable parameters (100) without tmm heads: {trainable_params_100}")
