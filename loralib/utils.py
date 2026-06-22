import math
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from .layers import HydraLinearLoRA, LoRALayer, PlainMultiheadAttentionLoRA

ADAPTER_PARAMETER_MARKERS = ('lora_', 'hydra_')
HYDRA_ADAPTATIONS = {'hydra'}


def is_adapter_parameter(name):
    return any(marker in name for marker in ADAPTER_PARAMETER_MARKERS)


def get_router_temperature_end(args):
    if getattr(args, 'router_temperature_schedule', 'fixed') == 'fixed':
        return args.router_temperature
    return getattr(args, 'router_temperature_end', args.router_temperature)


def get_router_temperature(args, step, total_steps):
    schedule = getattr(args, 'router_temperature_schedule', 'fixed')
    start = args.router_temperature
    end = get_router_temperature_end(args)
    if start <= 0 or end <= 0:
        raise ValueError('Router temperatures must be greater than 0')
    if schedule == 'fixed' or total_steps <= 1:
        return start

    progress = min(max(step / (total_steps - 1), 0.), 1.)
    if schedule == 'linear':
        return start + (end - start) * progress
    if schedule == 'cosine':
        weight = 0.5 * (1. + math.cos(math.pi * progress))
        return end + (start - end) * weight
    raise ValueError(f'Unknown router temperature schedule: {schedule}')


def set_hydra_temperature(list_lora_layers, temperature):
    if temperature <= 0:
        raise ValueError('Router temperature must be greater than 0')
    for layer in list_lora_layers:
        for module in layer.modules():
            if isinstance(module, HydraLinearLoRA):
                module.router_temperature = temperature


@contextmanager
def disabled_adapters(list_lora_layers):
    modules = []
    previous_states = []
    for layer in list_lora_layers:
        for module in layer.modules():
            if hasattr(module, 'adapters_disabled'):
                modules.append(module)
                previous_states.append(module.adapters_disabled)
                module.adapters_disabled = True
    try:
        yield
    finally:
        for module, previous_state in zip(modules, previous_states):
            module.adapters_disabled = previous_state


def get_adapter_metadata(args):
    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position,
        'adaptation': args.adaptation,
        'setting': getattr(args, 'setting', 'standard'),
    }
    if args.adaptation in HYDRA_ADAPTATIONS:
        metadata.update({
            'num_experts': args.num_experts,
            'router_temperature': args.router_temperature,
            'hydra_diversity_weight': getattr(args, 'hydra_diversity_weight', 0.),
            'hydra_balance_weight': getattr(args, 'hydra_balance_weight', 0.),
            'image_anchor_weight': getattr(args, 'image_anchor_weight', 0.),
            'text_anchor_weight': getattr(args, 'text_anchor_weight', 0.),
        })
    if args.adaptation in HYDRA_ADAPTATIONS:
        metadata.update({
            'router_temperature_schedule': getattr(
                args, 'router_temperature_schedule', 'fixed'),
            'router_temperature_end': get_router_temperature_end(args),
        })
    return metadata


def get_adapter_dir(args):
    if args.adaptation == 'hydra':
        if getattr(args, 'router_temperature_schedule', 'fixed') != 'fixed':
            return 'hydra_annealed_regularized'
        if (getattr(args, 'hydra_diversity_weight', 0.) > 0
                or getattr(args, 'hydra_balance_weight', 0.) > 0
                or getattr(args, 'image_anchor_weight', 0.) > 0
                or getattr(args, 'text_anchor_weight', 0.) > 0):
            return 'hydra_regularized'
    return args.adaptation


def get_adapter_save_dir(args):
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = (
        f'{args.save_path}/{backbone}/{args.dataset}/'
        f'{args.shots}shots/seed{args.seed}')
    setting = getattr(args, 'setting', 'standard')
    if setting != 'standard':
        save_dir = f'{save_dir}/{setting}'
    if args.adaptation != 'lora':
        save_dir = f'{save_dir}/{get_adapter_dir(args)}'
    return save_dir


INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        p.requires_grad = is_adapter_parameter(n)
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if is_adapter_parameter(name) and param.requires_grad:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def hydra_regularization_losses(list_lora_layers):
    diversity_losses = []
    balance_losses = []
    for layer in list_lora_layers:
        for module in layer.modules():
            if not isinstance(module, HydraLinearLoRA) or module.r <= 0:
                continue

            experts = F.normalize(module.hydra_B.float().flatten(1), dim=1)
            cosine = experts @ experts.t()
            off_diagonal = ~torch.eye(
                module.num_experts, dtype=torch.bool, device=cosine.device)
            diversity_losses.append(cosine[off_diagonal].square().mean())

            if module.last_router_probs is not None:
                probs = module.last_router_probs.float()
                balance_losses.append(
                    module.num_experts * probs.square().sum() - 1.)

    if not diversity_losses:
        raise ValueError('Hydra regularization requires at least one Hydra adapter')

    diversity = torch.stack(diversity_losses).mean()
    balance = (
        torch.stack(balance_losses).mean()
        if balance_losses else diversity.new_zeros(()))
    return diversity, balance


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if getattr(args, 'rank', 0) == 0:
                print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate,
                            adaptation=args.adaptation,
                            num_experts=args.num_experts,
                            router_temperature=args.router_temperature)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if getattr(args, 'rank', 0) == 0:
                print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate,
                            adaptation=args.adaptation,
                            num_experts=args.num_experts,
                            router_temperature=args.router_temperature)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    if args.adaptation in HYDRA_ADAPTATIONS:
        weights = {}
        for i, layer in enumerate(list_lora_layers):
            weights[f'layer_{i}'] = {
                name: parameter.detach().cpu()
                for name, parameter in layer.named_parameters()
                if is_adapter_parameter(name)
            }
        return save_adapter_data(args, weights)

    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    return save_adapter_data(args, weights, get_adapter_metadata(args))


def save_adapter_data(args, weights, metadata=None):
    if metadata is None:
        metadata = get_adapter_metadata(args)
    save_data = {'weights': weights, 'metadata': metadata}
    save_dir = get_adapter_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    load_dir = get_adapter_save_dir(args)
    load_path = f'{load_dir}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path, map_location='cpu')

    metadata = loaded_data['metadata']
    stored_adaptation = metadata.get('adaptation', 'lora')
    if stored_adaptation != args.adaptation:
        raise ValueError(
            f"Adaptation mismatch: expected {args.adaptation}, found {stored_adaptation}")
    stored_setting = metadata.get('setting', 'standard')
    expected_setting = getattr(args, 'setting', 'standard')
    if stored_setting != expected_setting:
        raise ValueError(
            f"Setting mismatch: expected {expected_setting}, found {stored_setting}")
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")
    if args.adaptation in HYDRA_ADAPTATIONS:
        if metadata['num_experts'] != args.num_experts:
            raise ValueError(
                f"Expert count mismatch: expected {args.num_experts}, found {metadata['num_experts']}")
        if metadata['router_temperature'] != args.router_temperature:
            raise ValueError(
                f"Router temperature mismatch: expected {args.router_temperature}, "
                f"found {metadata['router_temperature']}")
        stored_diversity = metadata.get('hydra_diversity_weight', 0.)
        stored_balance = metadata.get('hydra_balance_weight', 0.)
        stored_image_anchor = metadata.get('image_anchor_weight', 0.)
        stored_text_anchor = metadata.get('text_anchor_weight', 0.)
        expected_diversity = getattr(args, 'hydra_diversity_weight', 0.)
        expected_balance = getattr(args, 'hydra_balance_weight', 0.)
        expected_image_anchor = getattr(args, 'image_anchor_weight', 0.)
        expected_text_anchor = getattr(args, 'text_anchor_weight', 0.)
        if stored_diversity != expected_diversity:
            raise ValueError(
                f"Hydra diversity weight mismatch: expected {expected_diversity}, "
                f"found {stored_diversity}")
        if stored_balance != expected_balance:
            raise ValueError(
                f"Hydra balance weight mismatch: expected {expected_balance}, "
                f"found {stored_balance}")
        if stored_image_anchor != expected_image_anchor:
            raise ValueError(
                f"Image anchor weight mismatch: expected {expected_image_anchor}, "
                f"found {stored_image_anchor}")
        if stored_text_anchor != expected_text_anchor:
            raise ValueError(
                f"Text anchor weight mismatch: expected {expected_text_anchor}, "
                f"found {stored_text_anchor}")
        if args.adaptation == 'hydra':
            stored_schedule = metadata.get('router_temperature_schedule', 'fixed')
            expected_schedule = getattr(args, 'router_temperature_schedule', 'fixed')
            if stored_schedule != expected_schedule:
                raise ValueError(
                    f"Router temperature schedule mismatch: expected {expected_schedule}, "
                    f"found {stored_schedule}")
            stored_end = metadata.get(
                'router_temperature_end', metadata['router_temperature'])
            expected_end = get_router_temperature_end(args)
            if stored_end != expected_end:
                raise ValueError(
                    f"Final router temperature mismatch: expected {expected_end}, "
                    f"found {stored_end}")
        weights = loaded_data['weights']
        for i, layer in enumerate(list_lora_layers):
            parameters = dict(layer.named_parameters())
            for name, value in weights[f'layer_{i}'].items():
                parameters[name].data.copy_(value)
        print(f'LoRA weights loaded from {load_path}')
        if args.adaptation == 'hydra':
            inference_temperature = metadata.get(
                'router_temperature_end', metadata['router_temperature'])
            set_hydra_temperature(list_lora_layers, inference_temperature)
            print(f'Inference router temperature: {inference_temperature}')
        return

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
