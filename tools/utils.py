import torch


def load_model(model, model_path):
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # 使用字典推导式一次性处理state_dict
    filtered_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k
    }
    model.load_state_dict(filtered_dict)
    model.eval()
    return model
