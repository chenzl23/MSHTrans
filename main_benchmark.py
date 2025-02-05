import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
sys.path.append("./")
import logging
from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
import torch
from networks.MSHTrans import MSHTrans
import numpy as np
import random

seed_everything()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="MSHTran")
    parser.add_argument("--normalize", type=str, default="minmax")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--csv-postfix", type=str, default="common")
    parser.add_argument("--result-dir", type=str, default="./experimental_results", help="experimental results dir")
    parser.add_argument("--train-method", type=str, default="train_per_entity")
    parser.add_argument("--window-direction", type=str, default="backword")

    parser.add_argument("--dataset-id", type=str, default="SWaT")
    parser.add_argument("--data-root", type=str, default="./benchmark_data", help="dataset root")
    parser.add_argument("--train-postfix", type=str, default="train.pkl")
    parser.add_argument("--train-label-postfix", type=str, default="train_label.pkl")
    parser.add_argument("--test-postfix", type=str, default="test.pkl")
    parser.add_argument("--test-label-postfix", type=str, default="test_label.pkl")
    parser.add_argument("--n-feats", type=int, default=9)
    parser.add_argument("--valid-ratio", type=float, default=0)
    parser.add_argument("--entities", type=str, default="all")

    parser.add_argument("--metrics", nargs='+', type=str, default=["f1"])
    parser.add_argument("--best-target-metric", type=str, default="f1")
    parser.add_argument("--best-target-direction", type=str, default="max")
    parser.add_argument("--thresholding", nargs='+', type=str, default=["best"])
    parser.add_argument('--point-adjustment', type=bool, nargs=2, default=[True, False])


    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=5, help="K for top-k selection in hyperedges")
    parser.add_argument('--window-size', type=int, default=100, help='Window size')
    parser.add_argument('--pool-size-list', nargs='+', type=int, default=[2, 2], help='AvgPool size list')
    parser.add_argument("--nb-epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--update-rate", type=float, default=0.3, help="Rate for hyperedge updating during training")

    parser.add_argument('--hyper-num', nargs='+', type=int, default=[50, 30, 20])
    parser.add_argument('--head-num', type=int, default=3)
    parser.add_argument('--inner-size', type=int, default=5)
    parser.add_argument('--scale-num', type=int, default=3)
    parser.add_argument('--d-model', type=int, default=64)
    
    

    
    args = vars(parser.parse_args())
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    
    
    args["expid"] = args["dataset_id"]
    args["csv_postfix"] = args["dataset_id"] + "_" + args["csv_postfix"]
    args, device = set_logger(args)
    

    if args["dataset_id"] == "SMD":
        args["n_feats"] = 38
    elif args["dataset_id"] == "SWaT":
        args["n_feats"] = 51
    elif args["dataset_id"] == "PSM":
        args["n_feats"] = 25
    elif args["dataset_id"] == "MSL":
        args["n_feats"] = 55
    elif args["dataset_id"] == "WADI":
        args["n_feats"] = 123
    elif args["dataset_id"] == "SMAP":
        args["n_feats"] = 25
    elif args["dataset_id"] == "synthetic":
        args["n_feats"] = 3


    args["data_root"] = os.path.join(os.path.join(args["data_root"], args["dataset_id"]), "pkls")



    args["entities"] = np.load(os.path.join(args["data_root"], 'index_info.npy'))



    logging.info(print_to_json(args))

    data_dict = load_dataset(
         data_root = args["data_root"],
         entities = args["entities"],
         valid_ratio = args["valid_ratio"],
         dim = args["n_feats"],
         test_label_postfix = args["test_label_postfix"],
         test_postfix = args["test_postfix"],
         train_postfix = args["train_postfix"],
     )

    pp = data_preprocess.preprocessor(model_root=args["model_root"])
    data_dict = pp.normalize(data_dict, method=args["normalize"])


    evaluator = Evaluator(
        metrics = args["metrics"], 
        thresholding = args["thresholding"],
        best_params = {"target_metric": args["best_target_metric"], "target_direction": args["best_target_direction"]},
        point_adjustment = args["point_adjustment"]
        )

    anomaly_predictions_list = [] 
    eval_results_list = []


    for entity in args["entities"]:
        logging.info("Fitting dataset: {}".format(entity))

        args["seq_len"] = args["window_size"]



        window_dict = data_preprocess.generate_multi_windows(
            data_dict,
            entity = entity,
            window_size = args["window_size"],
            stride = args["stride"],
        )


        windows = window_dict[entity]

        train_windows = windows["train_windows"]
        test_windows = windows["test_windows"]
        
        train_time_series = torch.from_numpy(data_dict[entity]["train"])[: , :].float()
        test_time_series = torch.from_numpy(data_dict[entity]["test"])[: , :].float()



        train_loader, _, test_loader = get_dataloaders(train_windows, test_windows, train_time_series, test_time_series, batch_size=args["batch_size"])

        tt = TimeTracker(nb_epoch=args["nb_epoch"])

        model = MSHTrans(
            args,
            device
        ).to(device)
        
 
        tt.train_start()

        model.train(args, train_loader)

        tt.train_end()

        train_anomaly_score, _, _ = model.predict_prob(train_loader)

        tt.test_start()
        test_anomaly_score, loss, pred = model.predict_prob(test_loader)
        tt.test_end()

        store_entity(
            args,
            entity,
            train_anomaly_score,
            test_anomaly_score,
            windows["test_label"][ : , -1],
            time_tracker=tt.get_data(),
        )
        model_save_path = os.path.join(args["model_root"], entity)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'adjs': model.multi_adpive_hypergraph.adjs
        }
        torch.save(checkpoint, os.path.join(model_save_path, "model.pth"))
        logging.info(f"Saving model for {entity} done at {model_save_path}.")

        anomaly_predictions, eval_results_single = evaluator.eval_exp_single(args["model_root"], entity)
        
        anomaly_predictions_list.append(anomaly_predictions)
        eval_results_list.append(eval_results_single)

    evaluator.eval_all_average(
        merge_folder=args["benchmark_dir"],
        anomaly_predictions_list = anomaly_predictions_list, 
        eval_results_list = eval_results_list,
        args=args,
    )