import torch

MODEL_URLS = {
    "rqi": "https://huggingface.co/shaolin999/rqi/resolve/main/rqi_weights.pt"
}

def load_pretrained(model, name="rqi"):
    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URLS[name],
        map_location="cpu"
    )

    model.load_state_dict(state_dict, strict=True)
    return model