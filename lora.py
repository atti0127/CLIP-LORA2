import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from utils import *

from loralib.utils import (
    HYDRA_ADAPTATIONS,
    disabled_adapters,
    mark_only_lora_as_trainable,
    apply_lora,
    get_adapter_metadata,
    get_adapter_save_dir,
    get_lora_parameters,
    get_router_temperature,
    get_router_temperature_end,
    hydra_regularization_losses,
    lora_state_dict,
    load_lora,
    save_lora,
    set_hydra_temperature,
)
from loralib import layers as lora_layers


class TrainingLogger:
    def __init__(self, args):
        self.enabled = (
            is_main_process(args)
            and args.save_path is not None
            and not args.eval_only)
        self.file = None
        self.start_time = time.time()
        if self.enabled:
            save_dir = get_adapter_save_dir(args)
            os.makedirs(save_dir, exist_ok=True)
            self.file = open(
                os.path.join(save_dir, 'training_log.jsonl'),
                'w', encoding='utf-8')

    def log(self, event, **metrics):
        if not self.enabled:
            return
        record = {
            'event': event,
            'elapsed_seconds': time.time() - self.start_time,
            **metrics,
        }
        record = self._json_safe(record)
        self.file.write(json.dumps(record, sort_keys=True) + '\n')
        self.file.flush()

    def _json_safe(self, value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class CLIPFeatureEncoder(nn.Module):
    """Routes trainable CLIP feature extraction through DDP.forward."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images=None, texts=None):
        outputs = {}
        if images is not None:
            outputs['image_features'] = self.clip_model.encode_image(images)
        if texts is not None:
            outputs['text_features'] = self.clip_model.encode_text(texts)
        return outputs


def is_main_process(args):
    return getattr(args, 'rank', 0) == 0


def shard_texts(args, texts):
    if not getattr(args, 'distributed', False):
        return texts, len(texts)

    num_texts = len(texts)
    chunk_size = (num_texts + args.world_size - 1) // args.world_size
    padded_size = chunk_size * args.world_size
    if padded_size > num_texts:
        texts = torch.cat([
            texts,
            texts[-1:].expand(padded_size - num_texts, -1),
        ])
    start = args.rank * chunk_size
    return texts[start:start + chunk_size], num_texts


def gather_text_features(args, local_features, num_texts):
    if not getattr(args, 'distributed', False):
        return local_features
    gathered = dist_nn.all_gather(local_features)
    return torch.cat(gathered, dim=0)[:num_texts]


def distributed_mean(args, value):
    value = value.detach().float()
    if getattr(args, 'distributed', False):
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= args.world_size
    return value.item()


def tensor_collection_norm(tensors):
    tensors = list(tensors)
    if not tensors:
        return torch.zeros((), device='cuda')
    squared_norm = sum(
        tensor.detach().float().square().sum() for tensor in tensors)
    return squared_norm.sqrt()


def feature_anchor_loss(adapted_features, frozen_features):
    cosine = F.cosine_similarity(
        adapted_features.float(), frozen_features.detach().float(), dim=-1)
    return 1. - cosine.clamp(-1., 1.).mean()


def adapter_diagnostics(list_lora_layers, adapter_parameters):
    expert_cosines = []
    router_norms = []
    router_usage_maxes = []
    router_usage_entropies = []

    with torch.no_grad():
        for layer in list_lora_layers:
            for module in layer.modules():
                if isinstance(module, lora_layers.HydraLinearLoRA):
                    if module.r <= 0:
                        continue
                    experts = F.normalize(
                        module.hydra_B.float().flatten(1), dim=1)
                    cosine = experts @ experts.t()
                    off_diagonal = ~torch.eye(
                        module.num_experts, dtype=torch.bool,
                        device=cosine.device)
                    expert_cosines.append(
                        cosine[off_diagonal].square().mean())
                    router_norms.append(module.hydra_router.float().norm())
                    if module.last_router_probs is not None:
                        usage = module.last_router_probs.float()
                        router_usage_maxes.append(usage.max())
                        router_usage_entropies.append(
                            -(usage * usage.clamp_min(1e-12).log()).sum()
                            / math.log(module.num_experts))

        device = adapter_parameters[0].device
        diagnostics = {
            'adapter_norm': tensor_collection_norm(adapter_parameters),
            'expert_cosine_sq': (
                torch.stack(expert_cosines).mean()
                if expert_cosines else torch.zeros((), device=device)),
            'router_norm': (
                torch.stack(router_norms).mean()
                if router_norms else torch.zeros((), device=device)),
            'router_usage_max': (
                torch.stack(router_usage_maxes).mean()
                if router_usage_maxes else torch.zeros((), device=device)),
            'router_usage_entropy': (
                torch.stack(router_usage_entropies).mean()
                if router_usage_entropies else torch.zeros((), device=device)),
        }
    return diagnostics


def evaluate_zero_shot(clip_model, loader, textual_features, logit_scale):
    clip_model.eval()
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            logits = logit_scale * image_features @ textual_features
            acc += cls_acc(logits, target) * len(logits)
            tot_samples += len(logits)
    return acc / tot_samples


def harmonic_mean(first, second):
    denominator = first + second
    return 0. if denominator == 0 else 2. * first * second / denominator


def resolve_test_loaders(args, test_loader):
    if getattr(args, 'setting', 'standard') != 'base2new':
        return test_loader, None
    if not isinstance(test_loader, (tuple, list)) or len(test_loader) != 2:
        raise ValueError(
            'base2new evaluation requires (base_loader, novel_loader)')
    return test_loader[0], test_loader[1]


def evaluate_lora(args, clip_model, loader, dataset, classnames=None):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        if classnames is None:
            classnames = dataset.classnames
        texts = [
            template.format(classname.replace('_', ' '))
            for classname in classnames
        ]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def evaluate_test_splits(args, clip_model, test_loader, dataset):
    base_loader, novel_loader = resolve_test_loaders(args, test_loader)
    if novel_loader is None:
        return {
            'test_accuracy': evaluate_lora(
                args, clip_model, base_loader, dataset,
                classnames=dataset.test_classnames),
        }

    base_accuracy = evaluate_lora(
        args, clip_model, base_loader, dataset,
        classnames=dataset.test_classnames)
    novel_accuracy = evaluate_lora(
        args, clip_model, novel_loader, dataset,
        classnames=dataset.test_new_classnames)
    return {
        'base_accuracy': base_accuracy,
        'novel_accuracy': novel_accuracy,
        'harmonic_mean': harmonic_mean(base_accuracy, novel_accuracy),
    }


def print_test_metrics(metrics, prefix='Final'):
    if 'novel_accuracy' in metrics:
        print(
            f"**** {prefix} base accuracy: {metrics['base_accuracy']:.2f}; "
            f"novel accuracy: {metrics['novel_accuracy']:.2f}; "
            f"harmonic mean: {metrics['harmonic_mean']:.2f}. ****\n")
    else:
        print(
            f"**** {prefix} test accuracy: "
            f"{metrics['test_accuracy']:.2f}. ****\n")


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    logger = TrainingLogger(args)
    test_base_loader, test_new_loader = resolve_test_loaders(args, test_loader)
    # Textual features
    if is_main_process(args):
        print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    if is_main_process(args):
        print("\nEvaluating zero-shot CLIP on the test set.")
        zero_shot_base = evaluate_zero_shot(
            clip_model, test_base_loader, textual_features, logit_scale)
        if test_new_loader is None:
            zero_shot_metrics = {'test_accuracy': zero_shot_base}
        else:
            novel_textual_features = clip_classifier(
                dataset.test_new_classnames, dataset.template, clip_model)
            zero_shot_novel = evaluate_zero_shot(
                clip_model, test_new_loader, novel_textual_features,
                logit_scale)
            zero_shot_metrics = {
                'base_accuracy': zero_shot_base,
                'novel_accuracy': zero_shot_novel,
                'harmonic_mean': harmonic_mean(
                    zero_shot_base, zero_shot_novel),
            }
        print_test_metrics(zero_shot_metrics, prefix='Zero-shot CLIP')
        logger.log('zero_shot', **zero_shot_metrics)

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        if is_main_process(args):
            load_lora(args, list_lora_layers)
            test_metrics = evaluate_test_splits(
                args, clip_model, test_loader, dataset)
            print_test_metrics(test_metrics, prefix='Loaded adapter')
        if getattr(args, 'distributed', False):
            dist.barrier()
        logger.close()
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    feature_encoder = CLIPFeatureEncoder(clip_model)
    if getattr(args, 'distributed', False):
        feature_encoder = DistributedDataParallel(
            feature_encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    adapter_parameters = get_lora_parameters(feature_encoder)
    trainable_count = sum(parameter.numel() for parameter in adapter_parameters)
    if is_main_process(args):
        config_metadata = get_adapter_metadata(args)
        print(
            f"Adaptation: {args.adaptation}; "
            f"trainable adapter parameters: {trainable_count:,}")
        logger.log(
            'config',
            metadata=config_metadata,
            trainable_parameters=trainable_count,
            total_steps=total_iters,
            global_batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            per_gpu_microbatch_size=(
                args.batch_size
                // (args.world_size * args.accumulation_steps)),
            world_size=args.world_size,
            learning_rate=args.lr,
            log_interval=args.log_interval,
            val_interval=args.val_interval)
    optimizer = torch.optim.AdamW(adapter_parameters, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    optimizer_attempts = 0
    micro_steps = 0
    epoch = 0
    optimizer.zero_grad()
    while count_iters < total_iters:
        feature_encoder.train()
        if getattr(args, 'distributed', False):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        diversity_loss_epoch = 0.
        balance_loss_epoch = 0.
        image_anchor_loss_epoch = 0.
        text_anchor_loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(
                tqdm(train_loader, disable=not is_main_process(args))):
            micro_steps += 1
            optimizer_step = micro_steps % args.accumulation_steps == 0
            if (micro_steps - 1) % args.accumulation_steps == 0:
                accumulated_classification_loss = 0.
                accumulated_total_loss = 0.
                accumulated_diversity_loss = 0.
                accumulated_balance_loss = 0.
                accumulated_image_anchor_loss = 0.
                accumulated_text_anchor_loss = 0.
                accumulated_batch_accuracy = 0.
            if getattr(args, 'distributed', False):
                feature_encoder.require_backward_grad_sync = optimizer_step

            if args.adaptation == 'hydra':
                router_temperature = get_router_temperature(
                    args, count_iters, total_iters)
                set_hydra_temperature(list_lora_layers, router_temperature)

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            tokenized_texts = None
            num_texts = None
            if args.encoder == 'text' or args.encoder == 'both':
                tokenized_texts = clip.tokenize(texts).cuda()
                tokenized_texts, num_texts = shard_texts(args, tokenized_texts)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                encoded = feature_encoder(
                    images=images if args.encoder in {'vision', 'both'} else None,
                    texts=tokenized_texts)

            if args.encoder == 'text' or args.encoder == 'both':
                class_embeddings = gather_text_features(
                    args, encoded['text_features'], num_texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                image_features = encoded['image_features']
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            image_anchor_loss_value = image_features.new_zeros(())
            text_anchor_loss_value = image_features.new_zeros(())
            needs_image_anchor = (
                getattr(args, 'image_anchor_weight', 0.) > 0
                and args.encoder in {'vision', 'both'})
            needs_text_anchor = (
                getattr(args, 'text_anchor_weight', 0.) > 0
                and args.encoder in {'text', 'both'})
            if needs_image_anchor or needs_text_anchor:
                with torch.no_grad(), disabled_adapters(list_lora_layers):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        frozen_encoded = feature_encoder(
                            images=images if needs_image_anchor else None,
                            texts=tokenized_texts if needs_text_anchor else None)
                    if needs_image_anchor:
                        frozen_image_features = frozen_encoded['image_features']
                        frozen_image_features = (
                            frozen_image_features
                            / frozen_image_features.norm(dim=-1, keepdim=True))
                    if needs_text_anchor:
                        frozen_class_embeddings = gather_text_features(
                            args, frozen_encoded['text_features'], num_texts)
                        frozen_text_features = (
                            frozen_class_embeddings
                            / frozen_class_embeddings.norm(dim=-1, keepdim=True))
                if needs_image_anchor:
                    image_anchor_loss_value = feature_anchor_loss(
                        image_features, frozen_image_features)
                if needs_text_anchor:
                    text_anchor_loss_value = feature_anchor_loss(
                        text_features, frozen_text_features)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            classification_loss = F.cross_entropy(cosine_similarity, target)
            diversity_loss = classification_loss.new_zeros(())
            balance_loss = classification_loss.new_zeros(())
            if args.adaptation in HYDRA_ADAPTATIONS:
                diversity_loss, balance_loss = (
                    hydra_regularization_losses(list_lora_layers))
            loss = (
                classification_loss
                + args.hydra_diversity_weight * diversity_loss
                + args.hydra_balance_weight * balance_loss
                + getattr(args, 'image_anchor_weight', 0.) * image_anchor_loss_value
                + getattr(args, 'text_anchor_weight', 0.) * text_anchor_loss_value)
            batch_accuracy = (
                cosine_similarity.argmax(dim=1).eq(target).float().mean()
                * 100.)
            accumulation_scale = 1. / args.accumulation_steps
            accumulated_classification_loss += (
                classification_loss.detach() * accumulation_scale)
            accumulated_total_loss += loss.detach() * accumulation_scale
            accumulated_diversity_loss += (
                diversity_loss.detach() * accumulation_scale)
            accumulated_balance_loss += (
                balance_loss.detach() * accumulation_scale)
            accumulated_image_anchor_loss += (
                image_anchor_loss_value.detach() * accumulation_scale)
            accumulated_text_anchor_loss += (
                text_anchor_loss_value.detach() * accumulation_scale)
            accumulated_batch_accuracy += (
                batch_accuracy.detach() * accumulation_scale)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            diversity_loss_epoch += diversity_loss.item() * target.shape[0]
            balance_loss_epoch += balance_loss.item() * target.shape[0]
            image_anchor_loss_epoch += (
                image_anchor_loss_value.item() * target.shape[0])
            text_anchor_loss_epoch += (
                text_anchor_loss_value.item() * target.shape[0])
            tot_samples += target.shape[0]
            scaler.scale(loss / args.accumulation_steps).backward()
            if not optimizer_step:
                continue

            step = count_iters + 1
            scheduled_log = (
                step == 1 or step == total_iters
                or step % args.log_interval == 0)
            scaler.unscale_(optimizer)
            gradient_norm = tensor_collection_norm([
                parameter.grad for parameter in adapter_parameters
                if parameter.grad is not None
            ])
            applied_learning_rate = optimizer.param_groups[0]['lr']
            grad_scale_before = scaler.get_scale()
            optimizer_attempts += 1
            scaler.step(optimizer)

            scaler.update()
            grad_scale_after = scaler.get_scale()
            optimizer_step_skipped = grad_scale_after < grad_scale_before
            if not optimizer_step_skipped:
                scheduler.step()
                count_iters = step
            optimizer.zero_grad()

            should_log = scheduled_log or optimizer_step_skipped
            if should_log:
                diagnostics = adapter_diagnostics(
                    list_lora_layers, adapter_parameters)
                current_temperature = (
                    get_router_temperature(args, count_iters - 1, total_iters)
                    if args.adaptation == 'hydra'
                    else args.router_temperature)
                step_metrics = {
                    'step': count_iters,
                    'optimizer_attempt': optimizer_attempts,
                    'epoch': epoch,
                    'learning_rate': applied_learning_rate,
                    'router_temperature': current_temperature,
                    'classification_loss': distributed_mean(
                        args, accumulated_classification_loss),
                    'total_loss': distributed_mean(
                        args, accumulated_total_loss),
                    'diversity_loss': distributed_mean(
                        args, accumulated_diversity_loss),
                    'balance_loss': distributed_mean(
                        args, accumulated_balance_loss),
                    'image_anchor_loss': distributed_mean(
                        args, accumulated_image_anchor_loss),
                    'text_anchor_loss': distributed_mean(
                        args, accumulated_text_anchor_loss),
                    'batch_accuracy': distributed_mean(
                        args, accumulated_batch_accuracy),
                    'gradient_norm': distributed_mean(args, gradient_norm),
                    'gradient_finite': distributed_mean(
                        args, torch.isfinite(gradient_norm).float()),
                    'grad_scale_before': grad_scale_before,
                    'grad_scale_after': grad_scale_after,
                    'optimizer_step_skipped': optimizer_step_skipped,
                }
                step_metrics.update({
                    name: distributed_mean(args, value)
                    for name, value in diagnostics.items()
                })
                logger.log('train_step', **step_metrics)

            should_validate = (
                not optimizer_step_skipped
                and args.val_interval > 0
                and count_iters < total_iters
                and count_iters % args.val_interval == 0)
            if should_validate:
                if getattr(args, 'distributed', False):
                    dist.barrier()
                if is_main_process(args):
                    validation_accuracy = evaluate_lora(
                        args, clip_model, val_loader, dataset)
                    print(
                        f"**** Step {count_iters} validation accuracy: "
                        f"{validation_accuracy:.2f}. ****")
                    logger.log(
                        'periodic_validation',
                        step=count_iters,
                        validation_accuracy=validation_accuracy)
                if getattr(args, 'distributed', False):
                    dist.barrier()
                feature_encoder.train()
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters and is_main_process(args):
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            diversity_loss_epoch /= tot_samples
            balance_loss_epoch /= tot_samples
            image_anchor_loss_epoch /= tot_samples
            text_anchor_loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            current_temperature = (
                get_router_temperature(args, count_iters - 1, total_iters)
                if args.adaptation == 'hydra'
                else args.router_temperature)
            print(
                'LR: {:.6f}, Temp: {:.4f}, Acc: {:.4f}, Loss: {:.4f}, '
                'Diversity: {:.4f}, Balance: {:.6f}, ImageAnchor: {:.6f}, '
                'TextAnchor: {:.6f}'
                .format(
                    current_lr, current_temperature, acc_train, loss_epoch,
                    diversity_loss_epoch, balance_loss_epoch,
                    image_anchor_loss_epoch, text_anchor_loss_epoch))

    if is_main_process(args):
        if args.adaptation == 'hydra':
            set_hydra_temperature(
                list_lora_layers, get_router_temperature_end(args))

        test_metrics = evaluate_test_splits(
            args, clip_model, test_loader, dataset)
        print_test_metrics(test_metrics)
        logger.log('final', **test_metrics)

        if args.save_path != None:
            save_lora(args, list_lora_layers)
    if getattr(args, 'distributed', False):
        dist.barrier()
    logger.close()
    return
            
    
            
