# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect

import matplotlib.pyplot as plt
import numpy as np
import torch

from privacy_meter.audit import Audit
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource

import time
from privacy_meter.metric import PopulationMetric
from privacy_meter.information_source_signal import ModelLoss, ModelGradientNorm
import privacy_meter.hypothesis_test as prtest


class PM_report:  # NOQA: N801
    """
    This class indicates the information about the auditing and
    save the history of the privacy loss during the training.
    """

    def __init__(
        self,
        fpr_tolerance_list: list,
        is_report_roc: bool,
        signals: list,
        level: str,
        log_dir: str,
        interval: int = 1,
        other_info: dict = None,
    ) -> None:
        """
        Constructor

        Args:
            fpr_tolerance_list: FPR tolerance value(s) to be used by the audit.
            is_report_roc: Indicate whether the report should include the whole ROC
                           of the privacy loss
            signal: A list which indicates the auditing signal, e.g., [loss, gradient norm].
            level: Indicate which model to audit, e.g., local model or global model
            log_dir: Indicate where to save the privacy loss profile during the training.
            interval: Indicate the auditing interval
            other_info: contains other parameters, e.g., on which layer we want to
                        compute the gradient norm
        """
        self.fpr_tolerance = fpr_tolerance_list
        self.is_report_roc = is_report_roc
        self.signals = signals
        self.level = level  # which target model to audit
        self.log_dir = log_dir
        self.interval = interval
        self.history = {}  # history information based on each snapshot of the model
        self.other_info = other_info

        for attr in ["tpr", "fpr", "auc", "roc", "round"]:
            self.history[attr] = []

    def update_history(self, attr, info):
        self.history[attr].append(info)


def PopulationAuditor(target_model, datasets, pm_info):  # NOQA: N802
    """
    Function that returns updated privacy risk report based on the current
    snapshot of the model and FL history and updates the PM history.

    Args:
        target_model (PM model obj): The current snapshot of the model
        datasets (dict): Dataset dictionary, which contains members dataset and i
                         non-members dataset, as well as the dataset for auditing.
                         Each dataset is the PM dataset obj
        pm_info (PM_report obj): Dictionary that contains the history
                                 of the privacy loss report
    Returns:
        Updated privacy report
    """

    begin_time = time.time()

    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    population_dataset = datasets["audit"]

    # prepare for the dataset
    if torch.is_tensor(train_dataset.data):
        train_ds = {"x": train_dataset.data, "y": train_dataset.targets}
        test_ds = {"x": test_dataset.data, "y": test_dataset.targets}
        population_ds = {"x": population_dataset.data, "y": population_dataset.targets}
    else:
        train_ds = {
            "x": torch.from_numpy(train_dataset.data).float(),
            "y": torch.tensor(train_dataset.targets),
        }
        test_ds = {
            "x": torch.from_numpy(test_dataset.data).float(),
            "y": torch.tensor(test_dataset.targets),
        }
        population_ds = {
            "x": torch.from_numpy(population_dataset.data).float(),
            "y": torch.tensor(population_dataset.targets),
        }

    target_dataset = Dataset(
        data_dict={"train": train_ds, "test": test_ds},
        default_input="x",
        default_output="y",
        default_group="y",
    )

    # create the reference dataset
    pm_population_dataset = Dataset(
        data_dict={"train": population_ds},
        default_input="x",
        default_output="y",
        default_group="y",
    )

    target_info_source = InformationSource(
        models=[target_model], datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=[target_model], datasets=[pm_population_dataset]
    )

    print(f"debug:prepare for the dataset {time.time() - begin_time}")

    start_time = time.time()
    metrics = []
    for signal in pm_info.signals:
        if signal == "loss":
            metrics.append(
                PopulationMetric(
                    target_info_source=target_info_source,
                    reference_info_source=reference_info_source,
                    signals=[ModelLoss()],
                    hypothesis_test_func=prtest.linear_itp_threshold_func,
                )
            )
        elif signal == "logits":
            metrics.append(
                PopulationMetric(
                    target_info_source=target_info_source,
                    reference_info_source=reference_info_source,
                    signals=[ModelLoss()],
                    hypothesis_test_func=prtest.logit_rescale_threshold_func,
                )
            )
        elif signal == "gradient_norm":
            metrics.append(
                PopulationMetric(
                    target_info_source=target_info_source,
                    reference_info_source=reference_info_source,
                    signals=[
                        ModelGradientNorm(
                            pm_info.other_info["is_features"],
                            pm_info.other_info["layer_number"],
                        )
                    ],
                    hypothesis_test_func=prtest.linear_itp_threshold_func,
                )
            )
        else:
            raise ValueError(f"The provided signal {signal} is not supported.")

    print(f"debug:construct the metrics uses {time.time() - start_time}")

    start_time = time.time()
    audit_obj = Audit(
        metrics=metrics,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=None,
        save_logs=False,
    )
    audit_obj.prepare()
    print(f"preparing the auditing uses {time.time() - start_time}")

    start_time = time.time()
    audit_results = audit_obj.run()
    print(f"debug:running the auditing uses {time.time() - start_time}")

    start_time = time.time()
    roc_list = []
    auc_list = []
    t_tpr_dict = []
    t_fpr_dict = []
    for idx in range(len(audit_results)):
        # for each results in the signals
        mr = audit_results[idx][0]
        tpr_list = mr.tp / (mr.tp + mr.fn + 0.0000001)
        fpr_list = mr.fp / (mr.tn + mr.fp + 0.0000001)
        roc_list.append({"fpr": fpr_list, "tpr": tpr_list})
        auc_list.append(np.trapz(x=fpr_list, y=tpr_list))

        # focus on the performance uses are interested in
        tolerance_index = []
        for t_fpr in pm_info.fpr_tolerance:
            p = bisect(fpr_list, t_fpr)
            if p < len(fpr_list):
                tolerance_index.append(p)
            else:
                tolerance_index.append(p - 1)  # avoid to select the end point
        t_tpr_list = tpr_list[tolerance_index]
        t_fpr_list = fpr_list[tolerance_index]
        t_tpr_dict.append(t_tpr_list)
        t_fpr_dict.append(t_fpr_list)

    pm_info.update_history("tpr", t_tpr_dict)
    pm_info.update_history("fpr", t_fpr_dict)

    pm_info.update_history("auc", auc_list)
    pm_info.update_history("roc", roc_list)

    print(f"debug:saving the information uses {time.time() - start_time}")
    return pm_info


def plot_tpr_history(history_dict, client, fpr_tolerance_list):
    """
    Plot the history of true positive rate for each fpr tolerance
    in the fpr_tolerance_list  during the training.

    history_list: contains all the signals or reports we aim to show
    client: the auditing client
    name_list: idx for the history list
    fpr_tolerance_list: indicate the fpr tolerance list
    """
    for idx in range(len(fpr_tolerance_list)):
        for key in history_dict:
            for sidx, signal in enumerate(history_dict[key].signals):
                hist = np.array(history_dict[key].history["tpr"])[:, sidx, idx]
                plt.plot(
                    history_dict[key].history["round"], hist, label=f"{key}-{signal}"
                )
        plt.title(f"{client} (FPR @ {fpr_tolerance_list[idx]})")
        plt.legend()
        plt.savefig(
            f"{history_dict[key].log_dir}/{client}_tpr_at_{fpr_tolerance_list[idx]}.png",
            dpi=200,
        )
        plt.clf()


def plot_auc_history(history_dict, client):
    """
    Plot the history of a metric during the training.
    """

    for key in history_dict:
        for sidx, signal in enumerate(history_dict[key].signals):
            plt.plot(
                history_dict[key].history["round"],
                np.array(history_dict[key].history["auc"])[:, sidx],
                label=f"{key}-{signal}",
            )

    plt.title(f"{client} (AUC)")
    plt.legend()
    plt.savefig(f"{history_dict[key].log_dir}/{client}_auc.png", dpi=200)
    plt.clf()


def plot_roc_history(history_dict, client):
    """
    Plot the history of a metric during the training.
    """

    for key in history_dict:
        for sidx, signal in enumerate(history_dict[key].signals):
            tpr = history_dict[key].history["roc"][-1][sidx]["tpr"]
            fpr = history_dict[key].history["roc"][-1][sidx]["fpr"]
            plt.plot(fpr, tpr, label=f"{key}-{signal}")

        round_num = history_dict[key].history["round"][-1]  # the current round number

    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Random guess")

    plt.title(f"ROC curve - {client}")
    plt.legend()
    plt.savefig(
        f"{history_dict[key].log_dir}/{client}_roc_at_{round_num}.png",
        dpi=200,
    )
    plt.clf()
