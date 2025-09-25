class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'
    llama3_8b = 'Llama-3.1-8B'
    qwen2_14b = 'Qwen2.5-14B'
    phi4 = 'phi-4'
    gemma2_27b = 'gemma-2-27b-it'
    # TODO: (Yunzhao) add new model type

    # TODO: (Yunzhao) add new model type
    @staticmethod
    def formalize_model_name(x):
        if x == ModelTypes.opt_13b:
            return 'facebook/opt-13b'
        if x == ModelTypes.opt_66b:
            return 'facebook/opt-66b'
        if x == ModelTypes.opt_175b:
            return 'facebook/opt-175b'
        if x == ModelTypes.llama3_8b:
            return 'meta-llama/Llama-3.1-8B'
        if x == ModelTypes.qwen2_14b:
            return 'Qwen/Qwen2.5-14B-Instruct'
        if x == ModelTypes.phi4:
            return 'microsoft/phi-4'
        if x == ModelTypes.gemma2_27b:
            return 'google/gemma-2-27b-it'
        raise ValueError(x)

    # TODO: (Yunzhao) add new model type
    @staticmethod
    def model_str_to_object(model):
        if model == 'opt_13b' or model == "facebook/opt-13b":
            return ModelTypes.opt_13b
        if model == 'opt_66b' or model == "facebook/opt-66b":
            return ModelTypes.opt_66b
        if model == 'opt_175b' or model == "facebook/opt-175b":
            return ModelTypes.opt_175b
        if model == 'llama3_8b' or model == 'meta-llama/Llama-3.1-8B':
            return ModelTypes.llama3_8b
        if model == 'qwen2_14b' or model == 'Qwen/Qwen2.5-14B-Instruct':
            return ModelTypes.qwen2_14b
        if model == 'phi4' or model == 'microsoft/phi-4':
            return ModelTypes.phi4
        if model == 'gemma2_27b' or model == 'google/gemma-2-27b-it':
            return ModelTypes.gemma2_27b
        raise ValueError(model)
