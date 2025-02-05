import os
import logging
import pandas as pd
import numpy as np
import json
from collections import defaultdict, namedtuple
from common.exp import json_pretty_dump
from ..utils import load_hdf5, load_json, print_to_json
from .metrics import compute_binary_metrics, compute_delay
from .point_adjustment import adjust_predicts
from .thresholding import best_th, eps_th, pot_th


def get_comb_key(thresholding, point_adjustment):
    return "{}{}".format(thresholding, "_adjusted" if point_adjustment else "")


def results2csv(results, filepath, params):
    columns = [
        "uptime",
        "dataset_id",
        "strategy",
        "exp_id",
        "model_id",
        "length",
        "f1_adjusted",
        "pc_adjusted",
        "rc_adjusted",
        "acc_adjusted",
        "f1",
        "pc",
        "rc",
        "acc",
        "delay",
        "train_time",
        "test_time",
        "nb_epoch",
        "nb_eval_entity",
        "nb_total_entity",
    ]

    columns = columns + ["batch_size", "d_model", "lr", "window_size", "update_rate", "k", "hyper_num", "pool_size_list"]


    filedir = os.path.dirname(filepath)
    os.makedirs(filedir, exist_ok=True)

    total_rows = []
    basic_info = {
        key: value for key, value in results.items() if not isinstance(value, dict)
    }

    for key, value in results.items():
        if isinstance(value, dict):
            row = {"strategy": key, **value, **basic_info}
            row["batch_size"] = params["batch_size"]
            row["d_model"] = params["d_model"]
            row["lr"] = params["lr"]
            row["window_size"] = params["window_size"]
            row["update_rate"] = params["update_rate"]
            row["k"] = params["k"]
            row["hyper_num"] = str(params["hyper_num"])
            row["pool_size_list"] = str(params["pool_size_list"])
            total_rows.append(row)

    if os.path.isfile(filepath):
        logging.info(f"File {filepath} exists, loading directly.")
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame()
    total_rows.extend(df.to_dict(orient="records"))
    pd.DataFrame(total_rows, columns=columns).to_csv(filepath, index=False)
    logging.info(f"Appended exp results to {filepath}.")


class Evaluator:
    """

    th (str): options: "best", "eps", "pot"
    """

    def __init__(
        self,
        metrics,
        thresholding="best",
        pot_params={"q": 1e-3, "level": 0.99, "dynamic": False},
        best_params={"target_metric": "f1", "target_direction": "max"},
        point_adjustment=False,
        reverse_score=False
    ):
        if isinstance(thresholding, str):
            thresholding = [thresholding]
        if isinstance(point_adjustment, str):
            point_adjustment = [point_adjustment]

        self.thresholding = thresholding
        self.metrics = metrics
        self.best_params = best_params
        self.pot_params = pot_params
        self.point_adjustment = point_adjustment
        self.reverse_score = reverse_score

    def score2pred(
        self,
        thresholding,
        anomaly_score,
        anomaly_label,
        train_anomaly_score=None,
        point_adjustment=False,
    ):
        if self.reverse_score:
            anomaly_score = -anomaly_score


        pred_results = {"anomaly_pred": None, "anomaly_pred_adjusted": None, "th": None}

        if thresholding == "best":
            th = best_th(
                anomaly_score,
                anomaly_label,
                point_adjustment=point_adjustment,
                **self.best_params,
            )
        if thresholding == "pot":
            th = pot_th(train_anomaly_score, anomaly_score, **self.pot_params)
        if thresholding == "eps":
            th = eps_th(train_anomaly_score, reg_level=1)
            

        anomaly_pred = (anomaly_score > th).astype(int)

        pred_results["anomaly_pred"] = anomaly_pred
        pred_results["th"] = float(th)
        if self.point_adjustment:
            pred_results["anomaly_pred_adjusted"] = adjust_predicts(
                anomaly_pred, anomaly_label
            )
        print()
        return pred_results
    
    def score2th(
        self,
        thresholding,
        test_anomaly_score,
        test_anomaly_label,
        train_anomaly_score=None,
        point_adjustment=False,
    ):
        if self.reverse_score:
            test_anomaly_score = - test_anomaly_score

        if thresholding == "best":
            th = best_th(
                test_anomaly_score,
                test_anomaly_label,
                point_adjustment=point_adjustment,
                **self.best_params,
            )
        if thresholding == "pot":
            th = pot_th(train_anomaly_score, test_anomaly_score, **self.pot_params)
        if thresholding == "eps":
            th = eps_th(train_anomaly_score, reg_level=1)
            

        return th

    def th2pred(
        self,
        threshole,
        test_anomaly_score,
        test_anomaly_label,
        point_adjustment
    ):
        if self.reverse_score:
            test_anomaly_score = - test_anomaly_score


        pred_results = {"anomaly_pred": None, "anomaly_pred_adjusted": None, "th": None}

        anomaly_pred = (test_anomaly_score >= threshole).astype(int)

        pred_results["anomaly_pred"] = anomaly_pred
        pred_results["th"] = float(threshole)
        if point_adjustment:
            pred_results["anomaly_pred_adjusted"] = adjust_predicts(
                anomaly_pred, test_anomaly_label
            )
        return pred_results

    def eval(
        self,
        anomaly_label,
        anomaly_score=None,
        train_anomaly_score=None,
    ):
        eval_results = {}
        for point_adjustment in self.point_adjustment:
            for thresholding in self.thresholding:
                eval_results_tmp = {}

                pred_results = self.score2pred(
                    thresholding,
                    anomaly_score,
                    anomaly_label,
                    train_anomaly_score,
                    point_adjustment,
                )
                eval_results_tmp["th"] = float(pred_results["th"])
                anomaly_pred = pred_results["anomaly_pred"]
                key = get_comb_key(thresholding, point_adjustment)
                eval_results_tmp.update(
                    self.cal_metrics(anomaly_pred, anomaly_label, point_adjustment,key)
                )

                
                eval_results[key] = eval_results_tmp
        return eval_results

    def cal_metrics(self, anomaly_pred, anomaly_label, point_adjustment, key = None, entity_folder=None):
        logging.info(
            "Pred pos {}/{}, Label pos {}/{}".format(
                anomaly_pred.sum(),
                anomaly_pred.shape[0],
                anomaly_label.sum(),
                anomaly_label.shape[0],
            )
        )
        eval_metrics = {"length": anomaly_pred.shape[0]}
        for metric in self.metrics:
            if metric in ["f1", "pc", "rc","acc"]:
                eval_metrics.update(
                    compute_binary_metrics(
                        anomaly_pred,
                        anomaly_label,
                        point_adjustment,
                        key,
                        entity_folder
                    )
                )
            if metric == "delay":
                eval_metrics["delay"] = float(compute_delay(anomaly_pred, anomaly_label))
        return eval_metrics
    
    def evaluate_predictions(self, anomaly_pred, anomaly_label):
        eval_metrics = {
            'false_alarms': 0,
            'true_alarms': {},
            'missed_alarms': 0
        }
        
        starts = np.where(np.diff(np.pad(anomaly_pred.astype(int), (1, 1), 'constant')) == 1)[0]
        ends = np.where(np.diff(np.pad(anomaly_pred.astype(int), (1, 1), 'constant')) == -1)[0]

        false_alarms = 0
        for start in starts:
            end_candidates = ends[ends >= start]
            if len(end_candidates) > 0:
                end = end_candidates[0]
                if np.all(anomaly_label[start:end] == 0):
                    eval_metrics['false_alarms'] += 1
            else:
                break

        
        gt_changes = np.where(np.diff(np.pad(anomaly_label.astype(int), (1, 1), 'constant')) != 0)[0]
        for i in range(0, len(gt_changes), 2):
            start, end = gt_changes[i], gt_changes[i+1]
            gt_alarm_index = int(start)
            pred_alarm_times = np.where(anomaly_pred[start:end] == 1)[0] + start
            if pred_alarm_times.size > 0:
                nearest_pred_alarm = pred_alarm_times[np.argmin(np.abs(pred_alarm_times - gt_alarm_index))]
                delay = nearest_pred_alarm - gt_alarm_index
                eval_metrics['true_alarms'][gt_alarm_index] = int(delay)
            else:
                eval_metrics['missed_alarms'] += 1
        return eval_metrics
    
    def eval_exp_single(self, exp_folder, entity):
        eval_results_single = defaultdict(dict)
        anomaly_predictions = {
            "anomaly_pred": defaultdict(list),
            "anomaly_label": defaultdict(list),
        }

        entity_folder = os.path.join(exp_folder, entity)
        entity_folder = os.path.join(exp_folder, entity)
        try:
            score_dict = load_hdf5(
                os.path.join(entity_folder, f"score_{entity}.hdf5")
            )
            time_track = load_json(os.path.join(entity_folder, "time.json"))
        except:
            logging.warn("Failed to load entity {}.".format(entity))

        eval_results_single["train_time"] = float(time_track["train_time"])
        eval_results_single["test_time"] = float(time_track["test_time"])
        eval_results_single["nb_epoch"] = int(time_track["nb_epoch"])
        eval_results_single["nb_eval_entity"] = 1


        with open(os.path.join(entity_folder, "label.json"),'w') as f:
            json.dump(score_dict["test_anomaly_label"].tolist(),f)

        thresholds = {}
        for point_adjustment in self.point_adjustment:
            for thresholding in self.thresholding:
                EvalKey = namedtuple("key", ["point_adjustment", "thresholding"])
                eval_key = EvalKey(point_adjustment, thresholding)
                threshold = self.score2th(
                    thresholding,
                    score_dict["test_anomaly_score"],
                    score_dict["test_anomaly_label"],
                    score_dict["train_anomaly_score"],
                    point_adjustment = point_adjustment  
                )

                pred_results = self.th2pred(
                    threshold,
                    score_dict["test_anomaly_score"],
                    score_dict["test_anomaly_label"],
                    point_adjustment = point_adjustment
                )

                key = get_comb_key(eval_key.thresholding, eval_key.point_adjustment)
                
                thresholds[key] = pred_results["th"]
                if eval_key.point_adjustment:
                    anomaly_predictions["anomaly_pred"][eval_key] = pred_results["anomaly_pred_adjusted"]
                    
                else:
                    anomaly_predictions["anomaly_pred"][eval_key] = pred_results["anomaly_pred"]
                
                anomaly_predictions["anomaly_label"][eval_key] = score_dict["test_anomaly_label"]
                

                eval_results_single[key]["th"] = pred_results["th"]
                eval_results_single[key].update(
                    self.cal_metrics(
                        pred_results["anomaly_pred"],
                        score_dict["test_anomaly_label"],
                        point_adjustment,
                        key,
                        entity_folder
                    )
                )
                if eval_key.point_adjustment:
                    eval_results_single[key].update(
                        self.evaluate_predictions(
                            pred_results["anomaly_pred_adjusted"],
                            score_dict["test_anomaly_label"]
                        )
                    )
                else:
                    eval_results_single[key].update(
                        self.evaluate_predictions(
                            pred_results["anomaly_pred"],
                            score_dict["test_anomaly_label"]
                        )
                    )
                if eval_key.point_adjustment:
                    with open(os.path.join(entity_folder, "{}.json".format(key)),'w') as f:
                        json.dump(pred_results['anomaly_pred_adjusted'].tolist(),f)
                else :
                    with open(os.path.join(entity_folder, "{}.json".format(thresholding)),'w') as f:
                        json.dump(pred_results['anomaly_pred'].tolist(),f)
        
        json_pretty_dump(
            eval_results_single,
            os.path.join(entity_folder, "eval_results.json"),
        )


        json_pretty_dump(thresholds, os.path.join(entity_folder, "thresholds.json"))

        return anomaly_predictions, eval_results_single
    
    def eval_all_average(self, merge_folder, anomaly_predictions_list, eval_results_list, args, is_weighted = False):
        eval_results = {
        "dataset_id": args["dataset_id"],
        "exp_id": args["expid"],
        "model_id": args["model_id"],
        "nb_total_entity": len(anomaly_predictions_list),
        "uptime": args["uptime"],
        }

        num_entity = len(anomaly_predictions_list)

        thresholding = args["thresholding"]
        point_adjustment = args["point_adjustment"]

        perf_dict = {}
        weight_dict = {}
        for thres in thresholding:
            for pa in point_adjustment:
                eval_keys = get_comb_key(thres, pa)
                perf_dict[eval_keys] = {}
                weight_dict[eval_keys] = {}
                if pa == True:
                    perf_dict[eval_keys]["f1_adjusted"] = 0
                    perf_dict[eval_keys]["pc_adjusted"] = 0
                    perf_dict[eval_keys]["rc_adjusted"] = 0
                    perf_dict[eval_keys]["acc_adjusted"] = 0
                else:
                    perf_dict[eval_keys]["f1"] = 0
                    perf_dict[eval_keys]["pc"] = 0
                    perf_dict[eval_keys]["rc"] = 0
                    perf_dict[eval_keys]["acc"] = 0
                perf_dict[eval_keys]["length"] = 0
                weight_dict[eval_keys]["weight"] = 0

        for i in range(num_entity):
             eval_result = eval_results_list[i]
             for thres in thresholding:
                for pa in point_adjustment:
                    eval_key = get_comb_key(thres, pa)
                    perf = eval_result[eval_key]
                    if is_weighted:
                        weight_dict[eval_keys]["weight"] = perf['length']
                    else:
                        weight_dict[eval_keys]["weight"] = 1
                    perf_dict[eval_key]["length"] += perf['length']
                    if pa == True:
                        perf_dict[eval_key]["f1_adjusted"] += (perf["f1_adjusted"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["pc_adjusted"] += (perf["pc_adjusted"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["rc_adjusted"] += (perf["rc_adjusted"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["acc_adjusted"] += (perf["acc_adjusted"] * weight_dict[eval_keys]["weight"])
                    else:
                        perf_dict[eval_key]["f1"] += (perf["f1"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["pc"] += (perf["pc"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["rc"] += (perf["rc"] * weight_dict[eval_keys]["weight"])
                        perf_dict[eval_key]["acc"] += (perf["acc"] * weight_dict[eval_keys]["weight"])
        
        for thres in thresholding:
            for pa in point_adjustment:
                eval_key = get_comb_key(thres, pa)
                if is_weighted:
                    total = perf_dict[eval_key]["length"]
                else:
                    total = num_entity
                if pa == True:
                    perf_dict[eval_key]["f1_adjusted"] /= total
                    perf_dict[eval_key]["pc_adjusted"] /= total
                    perf_dict[eval_key]["rc_adjusted"] /= total
                    perf_dict[eval_key]["acc_adjusted"] /= total
                else:
                    perf_dict[eval_key]["f1"] /= total
                    perf_dict[eval_key]["pc"] /= total
                    perf_dict[eval_key]["rc"] /= total
                    perf_dict[eval_key]["acc"] /= total
        
        eval_results.update(perf_dict)

        logging.info(print_to_json(eval_results))
        logging.info(
            "Evaluated {} entities.".format(num_entity)
        )
        results2csv(
            eval_results,
            os.path.join(merge_folder, args["dataset_id"], "bench_results_" + args["csv_postfix"] + ".csv"), args
        )


    def eval_all(self, merge_folder, anomaly_predictions_list, eval_results_list, args):
        eval_results = {
            "dataset_id": args["dataset_id"],
            "exp_id": args["expid"],
            "model_id": args["model_id"],
            "train_time": 0,
            "test_time": 0,
            "nb_epoch": 0,
            "nb_eval_entity": 0,
            "nb_total_entity": len(anomaly_predictions_list),
            "uptime": args["uptime"],
        }

        num_entity = len(anomaly_predictions_list)
        merge_dict = {
            "anomaly_pred": defaultdict(list),
            "anomaly_label": defaultdict(list),
        }

        for i in range(num_entity):
            anomaly_predictions = anomaly_predictions_list[i]

            for eval_key in anomaly_predictions["anomaly_pred"].keys():
                merge_dict["anomaly_pred"][eval_key].append(anomaly_predictions["anomaly_pred"][eval_key])
            for eval_key in anomaly_predictions["anomaly_label"].keys():
                merge_dict["anomaly_label"][eval_key].append(anomaly_predictions["anomaly_label"][eval_key])

        for eval_key in merge_dict["anomaly_pred"].keys():
            # exit(0)
            key = get_comb_key(eval_key.thresholding, eval_key.point_adjustment)
            print(key)

            pred_cat = np.concatenate(merge_dict["anomaly_pred"][eval_key])
            label_cat = np.concatenate(merge_dict["anomaly_label"][eval_key])
            eval_result_tmp = self.cal_metrics(
                pred_cat, label_cat, eval_key.point_adjustment
            )
            eval_result_tmp["length"] = pred_cat.shape[0]
            eval_results[key] = eval_result_tmp

        
        logging.info(print_to_json(eval_results))
        logging.info(
            "Evaluated {} entities.".format(num_entity)
        )
        results2csv(
            eval_results,
            os.path.join(merge_folder, args["dataset_id"], "bench_results_" + args["csv_postfix"] + ".csv"), args
        )




    