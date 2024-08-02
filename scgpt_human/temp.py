import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import traceback
import numpy as np

from anndata import AnnData
import scanpy as sc
from typing import List, Tuple, Dict, Optional
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler
from scgpt.loss import (
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.utils import eval_scib_metrics
import warnings
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 自己写一个pre-training的trainer
def prepare_data(
    tokenized_train,
    tokenized_valid,
    train_batch_labels,
    valid_batch_labels,
    config,
    epoch,
    train_celltype_labels=None,
    valid_celltype_labels=None,
    sort_seq_batch=False,
) -> Tuple[Dict[str, torch.Tensor]]:
    assert config.task in ["annotation", "integration", "perturb", "multiomic"]
    # 给训练集加掩码(train+valid)
    # 随机将mask_ratio部分的元素values替换成mask_value
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == config.mask_value).sum() / (masked_values_train - config.pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    # value加掩码
    input_values_train, input_values_valid = masked_values_train, masked_values_valid

    # value不加掩码
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    # 在深度学习中，标签数据通常需要是整数类型，特别是当使用交叉熵损失函数（如 torch.nn.CrossEntropyLoss）时，标签必须是长整型（torch.int64）。
    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    if config.task == "annotation":
        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if config.task == "multiomic":
        tensor_mod_types_train, tensor_mod_types_valid = (
            tokenized_train["mod_types"].long(),
            tokenized_valid["mod_types"].long(),
        )

    if sort_seq_batch:
        # 根据训练数据的批次标签对训练数据进行排序
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        if config.task == "annotation":
            tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]
        if config.task == "multiomic":
            tensor_mod_types_train = tensor_mod_types_train[train_sort_ids]

        # 根据验证数据的批次标签对验证数据进行排序
        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        if config.task == "annotation":
            tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]
        if config.task == "multiomic":
            tensor_mod_types_valid = tensor_mod_types_valid[valid_sort_ids]

    # 数据打包
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }
    if config.task == "annotation":
        train_data_pt["celltype_labels"] = tensor_celltype_labels_train
        valid_data_pt["celltype_labels"] = tensor_celltype_labels_valid
    if config.task == "multiomic":
        train_data_pt["mod_types"] = tensor_mod_types_train
        valid_data_pt["mod_types"] = tensor_mod_types_valid
    return train_data_pt, valid_data_pt


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # 根据batch_label_array将数据分成多个子集，在子集内和子集之间可以进行随机打乱
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def train(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    logger,
    epoch,
) -> None:
    """
    Train the model for one epoch.
    """
    import wandb

    model.train()   # 将模型设置为训练模式
    total_loss, total_gep, total_cls, total_gepc, total_ecs, total_dab = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    total_zero_log_prob, total_gepc_zero_log_prob = 0.0, 0.0
    # total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)   # 批次数量
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        # input_values部分被替换成了mask_value,而targt_values没有被替换
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        if config.task == "annotation":
            celltype_labels = batch_data["celltype_labels"].to(device)
        if config.task == "multiomic":
            mod_types = batch_data["mod_types"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

        # 自动混合精度训练
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels
                if config.use_batch_labels or config.DSBN
                else None,
                CLS=config.CLS,
                MVC=config.GEPC,
                ECS=config.ESC,
                # mod_types=mod_types if config.use_mod else None,
                # do_sample=do_sample_in_train,
                # generative_training=False
            )

            masked_positions = input_values.eq(
                config.mask_value
            )  # the postions to predict
            non_masked_positions = ~input_values.eq(config.mask_value)
            # gen_key_padding_mask = data_dict["gen_key_padding_mask"]
            loss = 0.0
            metrics_to_log = {}

            # 加上目标函数
            # loss_goal=criterion_cls(
            #     output_dict["mlm_output"], target_values, non_masked_positions
            # )
            # loss = loss + loss_goal

            # 下面都是精调的任务
            # gene expression prediction
            # 把GEP当做目标函数使用
            if config.GEP:
                loss_gep = criterion_gep_gepc(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_gep
                metrics_to_log = {"train/gep": loss_gep.item()}
                # print("loss of GEP: ",loss)   为什么是nan???
            # 为什么对于0概率要专门再算一遍，而且不给loss加权重？？
            if config.GEP and config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})

            # gene expression prediction for cell modeling
            if config.GEPC:
                loss_gepc = criterion_gep_gepc(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
                

            if config.GEPC and config.explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )

            # cell type classification
            if config.CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

            # elastic cell similarity
            if config.ESC:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})

            # domain adaptation via reverse back propagation
            if config.DAR:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_gep += loss_gep.item() if config.GEP else 0.0
        total_cls += loss_cls.item() if config.CLS else 0.0
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_ecs += loss_ecs.item() if config.ESC else 0.0
        total_dab += loss_dab.item() if config.DAR else 0.0
        total_zero_log_prob += (
            loss_zero_log_prob.item()
            if config.GEP and config.explicit_zero_prob
            else 0.0
        )
        total_gepc_zero_log_prob += (
            loss_gepc_zero_log_prob.item()
            if config.GEPC and config.explicit_zero_prob
            else 0.0
        )
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_cls = total_cls / log_interval if config.CLS else 0.0
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_ecs = total_ecs / log_interval if config.ESC else 0.0
            cur_dab = total_dab / log_interval if config.DAR else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if config.explicit_zero_prob else 0.0
            )
            cur_gepc_zero_log_prob = (
                total_gepc_zero_log_prob / log_interval
                if config.GEPC and config.explicit_zero_prob
                else 0.0
            )
            # cur_error = total_error / log_interval

            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"gep {cur_gep:5.2f} |" if config.GEP else "")
                + (f"cls {cur_cls:5.2f} | " if config.CLS else "")
                # + (f"err {cur_error:5.2f} | " if config.CLS else "")
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"ecs {cur_ecs:5.2f} |" if config.ESC else "")
                + (f"dar {cur_dab:5.2f} |" if config.DAR else "")
            )
            total_loss = 0
            total_gep = 0
            total_cls = 0
            total_gepc = 0
            total_ecs = 0
            total_dab = 0
            total_zero_log_prob = 0
            total_gepc_zero_log_prob = 0
            # total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    import wandb

    wandb.define_metric("valid/loss", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    device,
    config,
    epoch,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    import wandb

    model.eval()
    total_loss = 0.0
    # total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            if config.task == "annotation":
                celltype_labels = batch_data["celltype_labels"].to(device)
            if config.task == "multiomic":
                mod_types = batch_data["mod_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,  # evaluation does not need CLS or CCE
                    MVC=False,
                    ECS=False,
                    # mod_types=mod_types if config.use_mod else None,
                    # do_sample=do_sample_in_train,
                    # generative_training = False,
                )
                if config.task == "annotation":
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                elif config.task in ["integration", "multiomic"]:
                    output_values = output_dict["mlm_output"]
                    masked_positions = input_values.eq(config.mask_value)
                    loss = criterion_gep_gepc(
                        output_values, target_values, masked_positions
                    )

                if config.DAR:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)

            if config.DAR:
                total_dab += (
                    loss_dab.item() * len(input_gene_ids) if config.DAR else 0.0
                )
            else:
                total_dab = 0

            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/loss": (total_loss + config.dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num


def predict(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    config,
    device,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch,batch_data in enumerate(loader):
            try:
                # 检查 input_gene_ids 的范围和形状
                print("input_gene_ids shape:", batch_data["gene_ids"].shape)
                print("input_gene_ids max index:", batch_data["gene_ids"].max().item())
                print("input_gene_ids min index:", batch_data["gene_ids"].min().item())

                # 检查 input_values 的范围和形状
                print("input_values shape:", batch_data["values"].shape)

                # 检查 target_values 的范围和形状
                print("target_values shape:", batch_data["target_values"].shape)

                # 检查 batch_labels 的范围和形状
                print("batch_labels shape:", batch_data["batch_labels"].shape)
                print("batch_labels max index:", batch_data["batch_labels"].max().item())
                print("batch_labels min index:", batch_data["batch_labels"].min().item())
                # 转移数据到GPU
                # 要移到cuda上需要是long类型的，在执行数据迁移的时候，可能会自动调用tensor的索引
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                # input_gene_ids = batch_data["gene_ids"]
                # input_values = batch_data["values"]
                # target_values = batch_data["target_values"]
                # batch_labels = batch_data["batch_labels"]
                # celltype_labels = batch_data["celltype_labels"].to(device)
            except Exception as e:
                print(f"Error processing batch {batch}: {e}")
                print(f"Batch data: {batch_data}")
                raise e
            # celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,
                    MVC=False,
                    ECS=False,
                )

                # output_values = output_dict["cls_output"]
                # preds = output_values.argmax(1).cpu().numpy()
                # predictions.append(preds)

                output_values=output_dict["mlm_output"]
                predictions.append(output_values)

    return np.concatenate(predictions, axis=0)


# %% inference
def test(
    model: nn.Module, adata: DataLoader, gene_ids, vocab, config, device, logger
) -> float:
    all_counts = (
        adata.layers[config.input_layer_key].A
        if issparse(adata.layers[config.input_layer_key])
        else adata.layers[config.input_layer_key]
    )

    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    (
        test_data,
        _,
        test_celltype_labels,
        _,
        test_batch_labels,
        _,
    )=train_test_split(
        all_counts,celltypes_labels,batch_ids,test_size=0,shuffle=True
    )

    tokenized_test = tokenize_and_pad_batch(
        test_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=config.include_zero_gene,
    )

    # 加掩码
    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )

    # 正确答案
    target_values_test=tokenized_test["values"]

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(test_batch_labels).long(),
        # "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    # test_loader = DataLoader(
    #     dataset=SeqDataset(test_data_pt),
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=min(len(os.sched_getaffinity(0)), config.batch_size // 2),
    #     pin_memory=True,
    # )

    test_loader=prepare_dataloader(
        test_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )

    model.eval()
    # predictions = predict(
    #     model,
    #     test_loader,
    #     vocab,
    #     config,
    #     device,
    # )

    # masked_positions=input_values_test.eq(config.mask_value)
    # loss=criterion_gep_gepc(predictions,target_values_test,masked_positions)
    loss=0
    for epoch in range(1,config.epochs+1):
        loss=loss+evaluate(
            model,
            test_loader,
            vocab,
            masked_mse_loss,
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            device,
            config,
            epoch
        )
    
    # compute accuracy, precision, recall, f1
    # accuracy = accuracy_score(target_values_test, predictions)
    # precision = precision_score(celltypes_labels, predictions, average="macro")
    # recall = recall_score(celltypes_labels, predictions, average="macro")
    # macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    # micro_f1 = f1_score(celltypes_labels, predictions, average="micro")

    # logger.info(
    #     f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
    #     f"Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}"
    # )

    logger.info(
        f"Loss: {loss:.3f}"
    )

    results={
        "test/loss": loss,
    }

    # results = {
    #     "test/accuracy": accuracy,
    #     "test/precision": precision,
    #     "test/recall": recall,
    #     "test/macro_f1": macro_f1,
    #     "test/micro_f1": micro_f1,
    # }

    # return predictions, celltypes_labels, results
    return predictions,results


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids,
    vocab,
    config,
    logger,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[config.input_layer_key].A
        if issparse(adata_t.layers[config.input_layer_key])
        else adata_t.layers[config.input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=config.include_zero_gene,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long()
                if config.DSBN or config.DAR or config.use_batch_labels
                else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results
