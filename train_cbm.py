# train_cbm_multiseed_waterbirds.py
#
# Run Waterbirds 3 seeds, avoid save-folder collisions, and report mean/std of
# overall accuracy and worst-group accuracy.

import os
import json
import uuid
import torch
import random
import argparse
import datetime
import numpy as np
from collections import Counter

import utils
import data_utils        # must expose waterbirds_* and helpers incl. get_groups_only
import similarity

from torch.utils.data import DataLoader, TensorDataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

ALEXNET_ERRORS = {
    'gaussian_noise': 0.887, 'shot_noise': 0.879, 'impulse_noise': 0.855,
    'defocus_blur': 0.803, 'glass_blur': 0.817, 'motion_blur': 0.796,
    'zoom_blur': 0.832, 'snow': 0.844, 'frost': 0.819, 'fog': 0.816,
    'brightness': 0.715, 'contrast': 0.813, 'elastic_transform': 0.733,
    'pixelate': 0.805, 'jpeg_compression': 0.758
}

def compute_mce(per_corr_per_sev: dict, alexnet_ce: dict):
    """
    per_corr_per_sev: {corruption: {severity(int): {'acc': float, 'error': float}}}
    alexnet_ce:       {corruption: CE_baseline(float)}

    Returns: mce_percent, details_dict, included_corruptions
    """
    included = []
    details = {}
    nces = []

    for corr, sev_dict in per_corr_per_sev.items():
        if corr not in alexnet_ce:
            continue  # no AlexNet baseline -> skip
        # average error over available severities
        sevs = sorted(sev_dict.keys())
        errs = [(sev_dict[s]['error']) for s in sevs]
        ce = sum(errs) / max(1, len(errs))
        base = alexnet_ce[corr]
        nce = ce / base if base > 0 else float('nan')

        included.append(corr)
        details[corr] = {'CE': ce, 'AlexNet_CE': base, 'NCE': nce}

        if not (nce != nce):  # exclude NaN
            nces.append(nce)

    mce_percent = (sum(nces) / max(1, len(nces))) * 100.0
    return mce_percent, details, included


parser = argparse.ArgumentParser(description='Settings for creating CBM')

parser.add_argument("--dataset", type=str, default="cifar10c")
parser.add_argument("--concept_set", type=str, default="/home/eng/eisenbr2/Label-free-CBM-2/data/concept_sets/cifar10_filtered.txt",
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
parser.add_argument("--cifar10c_root", type=str,
                    default="/dsi/dsai-lab/Ran/CIFAR-10-C-P/CIFAR-10-C",
                    help="Path to CIFAR-10-C directory (with *.npy and labels.npy)")
parser.add_argument("--cifar10c_corruptions", type=str,
                    # default="gaussian_noise,shot_noise",
                    default="gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression,speckle_noise,gaussian_blur,spatter,saturate",
                    help="Comma-separated list of CIFAR-10-C corruptions")
parser.add_argument("--cifar10c_severities", type=str, default="1,2,3,4,5",
                    help="Comma-separated severities to include (1..5)")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4',
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='/dsi/dsai-lab/Ran/cbm/cifar10c/saved_activations',
                    help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='/dsi/dsai-lab/Ran/cbm/cifar10c/saved_models',
                    help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25,
                    help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000,
                    help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45,
                    help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007,
                    help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000,
                    help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")

# Run multiple seeds
parser.add_argument("--seeds", type=str, default="999",
                    help="Comma-separated list of seeds, e.g. '999,2024,7'")


def _make_unique_save_dir(base_dir: str, dataset: str, seed: int) -> str:
    """
    Create a unique save directory to avoid FileExistsError when runs happen in the same minute.
    Includes seconds + seed and, if needed, a short uuid suffix.
    """
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base = os.path.join(base_dir, f"{dataset}_cbm_{ts}_s{seed}")
    cand = base
    # Try once; if exists, append a short uuid
    try:
        os.makedirs(cand, exist_ok=False)
        return cand
    except FileExistsError:
        cand = f"{base}_{uuid.uuid4().hex[:6]}"
        os.makedirs(cand, exist_ok=False)
        return cand


def train_cbm_and_save(args):

    os.makedirs(args.save_dir, exist_ok=True)

    similarity_fn = similarity.cos_similarity_cubed_single

    d_train = args.dataset + "_train"
    d_val   = args.dataset + "_val"

    # get classes & concepts
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = [c for c in f.read().split("\n") if len(c) > 0]

    with open(args.concept_set) as f:
        concepts = [c for c in f.read().split("\n") if len(c) > 0]

    # save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(
            clip_name=args.clip_name, target_name=args.backbone,
            target_layers=[args.feature_layer], d_probe=d_probe,
            concept_set=args.concept_set, batch_size=args.batch_size,
            device=args.device, pool_mode="avg", save_dir=args.activation_dir
        )

    target_save_name, clip_save_name, text_save_name = utils.get_save_names(
        args.clip_name, args.backbone, args.feature_layer, d_train, args.concept_set, "avg", args.activation_dir
    )
    val_target_save_name, val_clip_save_name, text_save_name = utils.get_save_names(
        args.clip_name, args.backbone, args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir
    )

    # load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features

    # filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i] <= args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i] > args.clip_cutoff]

    # save memory by recalculating with filtered concepts
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest > args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        del image_features, text_features

    val_clip_features = val_clip_features[:, highest > args.clip_cutoff]

    # learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = list(range(len(target_features)))

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)

        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == args.proj_steps - 1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i == 0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(
                    best_step, -loss.cpu(), -best_val_loss.cpu()))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else:
                break
        opt.zero_grad()

    proj_layer.load_state_dict({"weight": best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    # delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff

    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i] <= args.interpretability_cutoff:
                print("Deleting {}, Interpretability:{:.3f}".format(concept, sim[i]))

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]

    del clip_features, val_clip_features

    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight": W_c})

    # targets (labels) from datasets
    train_targets = data_utils.get_targets_only(d_train)
    val_targets   = data_utils.get_targets_only(d_val)

    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c   = proj_layer(val_target_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std  = torch.std(train_c, dim=0, keepdim=True)

        # numerical safety
        denom = torch.clamp(train_std, min=1e-12)
        train_c = (train_c - train_mean) / denom
        val_c   = (val_c   - train_mean) / denom

        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(val_c, val_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    # Solve the GLM path
    output_proj = glm_saga(
        linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
        val_loader=val_loader, do_zero=False, metadata=metadata,
        n_ex=len(target_features), n_classes=len(classes)
    )
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    # ===========================
    # Evaluate Accuracies
    # ===========================
    val_groups = data_utils.get_groups_only(d_val)  # may be None for CIFAR-10-C

    linear.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x, y) in val_loader:
            x = x.to(args.device)
            logits = linear(x)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    # Overall accuracy
    overall_correct = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    overall_acc = overall_correct / max(1, len(all_labels))
    print("Overall Accuracy:", overall_acc)

    metrics_for_json = {"overall_accuracy": overall_acc,
                        "overall_error": 1.0 - overall_acc}
    
    # ===========================
    # Evaluate Accuracies
    # ===========================
    val_groups = data_utils.get_groups_only(d_val)  # may be None for CIFAR-10-C

    # Initialize to None so they're always defined
    group_acc = None
    worst_group_acc = None
    weighted_group_acc = None


    if val_groups is not None:
        # ======= Group metrics path (Waterbirds/Sparwious) =======
        group_acc = {}
        total_samples = 0
        weighted_correct = 0
        for g in sorted(set(val_groups)):
            idxs = [i for i, gid in enumerate(val_groups) if gid == g]
            if idxs:
                correct = sum(1 for i in idxs if all_preds[i] == all_labels[i])
                group_acc[g] = correct / len(idxs)
                weighted_correct += correct
                total_samples += len(idxs)
            else:
                group_acc[g] = float('nan')

        worst_group_acc = min(v for v in group_acc.values() if v == v)
        weighted_group_acc = weighted_correct / max(1, total_samples)

        print("Group-wise Accuracy:", group_acc)
        print("Worst Group Accuracy:", worst_group_acc)
        print("Weighted Group Accuracy:", weighted_group_acc)

        metrics_for_json.update({
            "group_metrics": {
                "group_accuracy": group_acc,
                "worst_group_accuracy": worst_group_acc,
                "weighted_group_accuracy": weighted_group_acc
            }
        })
    else:
        # ======= CIFAR-10-C path (no groups) =======
        # If available, pull segment info to compute per-corruption & per-severity metrics
        segments = None
        try:
            segments = data_utils.get_cifar10c_segments(d_val)
        except Exception:
            pass

        if segments:
            # Aggregate metrics
            from collections import defaultdict
            corr_totals = defaultdict(lambda: {"correct": 0, "count": 0})
            corr_sev = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "count": 0}))

            for seg in segments:
                s, e = seg["start"], seg["end"]
                corr, sev = seg["name"], int(seg["severity"])
                correct = sum(1 for i in range(s, e) if all_preds[i] == all_labels[i])
                cnt = e - s
                corr_totals[corr]["correct"] += correct
                corr_totals[corr]["count"] += cnt
                corr_sev[corr][sev]["correct"] += correct
                corr_sev[corr][sev]["count"] += cnt

            # Print nicely and prepare JSON
            per_corr = {}
            per_corr_sev = {}
            for corr, agg in corr_totals.items():
                acc = agg["correct"] / max(1, agg["count"])
                err = 1.0 - acc
                per_corr[corr] = {"acc": acc, "error": err}
                print(f"{corr:>20s}: acc={acc:.4f}, error={err:.4f}")

                per_corr_sev[corr] = {}
                for sev, a in sorted(corr_sev[corr].items()):
                    acc_s = a["correct"] / max(1, a["count"])
                    per_corr_sev[corr][int(sev)] = {"acc": acc_s, "error": 1.0 - acc_s}
                    print(f"  severity {sev}: acc={acc_s:.4f}, error={1.0 - acc_s:.4f}")

            metrics_for_json.update({
                "cifar10c_metrics": {
                    "per_corruption": per_corr,
                    "per_corruption_per_severity": per_corr_sev
                }
            })

            # --- mCE (AlexNet-normalized) ---
            mce_percent, mce_details, mce_corrs = compute_mce(per_corr_sev, ALEXNET_ERRORS)
            print(f"mCE (AlexNet-normalized over {len(mce_corrs)} corruptions): {mce_percent:.2f}%")
            for corr in sorted(mce_corrs):
                d = mce_details[corr]
                print(f"  {corr:>20s}: CE={d['CE']:.4f}, AlexNet_CE={d['AlexNet_CE']:.3f}, NCE={d['NCE']:.3f}")

            # stash in metrics for saving
            metrics_for_json.setdefault("cifar10c_metrics", {})
            metrics_for_json["cifar10c_metrics"]["mce_percent"] = float(mce_percent)
            metrics_for_json["cifar10c_metrics"]["mce_details"] = mce_details



    # ===========================
    # Save artifacts & metrics
    # ===========================
    # Use a unique directory (seconds + seed + optional uuid)
    save_name = _make_unique_save_dir(args.save_dir, args.dataset, seed=getattr(args, "seed", -1))

    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name, "W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0] if len(concepts) else "")
        for concept in concepts[1:]:
            f.write('\n' + concept)

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            try:
                out_dict[key] = float(output_proj['path'][0][key])
            except Exception:
                pass
        out_dict['metrics'] = output_proj['path'][0].get('metrics', {})

        nnz = (W_g.abs() > 1e-5).sum().item()
        total_w = W_g.numel()
        out_dict['sparsity'] = {
            "Non-zero weights": nnz,
            "Total weights": total_w,
            "Percentage non-zero": nnz / total_w
        }

        # Always present
        out_dict['overall_accuracy'] = metrics_for_json['overall_accuracy']
        out_dict['overall_error']    = metrics_for_json['overall_error']

        # Present only if groups exist (Waterbirds/Sparwious)
        if 'group_metrics' in metrics_for_json:
            out_dict['group_metrics'] = metrics_for_json['group_metrics']

        # Present only for CIFAR-10-C runs
        if 'cifar10c_metrics' in metrics_for_json:
            out_dict['cifar10c_metrics'] = metrics_for_json['cifar10c_metrics']

        json.dump(out_dict, f, indent=2)


    # Return metrics requested for multi-seed summary
    return worst_group_acc, overall_acc


if __name__ == '__main__':
    args = parser.parse_args()

    # Configure CIFAR-10-C (no-op for other datasets)
    try:
        corr_list = [c.strip() for c in args.cifar10c_corruptions.split(",") if c.strip()]
        sev_list  = [int(s.strip()) for s in args.cifar10c_severities.split(",") if s.strip()]
        data_utils.set_cifar10c_options(root=args.cifar10c_root,
                                        corruptions=corr_list, severities=sev_list)
    except Exception:
        pass


    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    worst_group_accs = []
    overall_accs = []

    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Keep the seed in args for saving and logging
        # (avoid mutating the original argparse.Namespace too much)
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed

        worst_acc, overall_acc = train_cbm_and_save(run_args)
        worst_group_accs.append(worst_acc)
        overall_accs.append(overall_acc)

    # Report mean and std across seeds
    print("\n====== Seed Summary ======")
    print(f"Seeds: {seeds}")
    print(f"Overall Accuracy:     mean = {np.mean(overall_accs):.4f}, std = {np.std(overall_accs):.4f}")
    if worst_group_accs:
        print(f"Worst Group Accuracy: mean = {np.mean(worst_group_accs):.4f}, std = {np.std(worst_group_accs):.4f}")
    else:
        print("Worst Group Accuracy: (not applicable — no groups in CIFAR-10-C)")
