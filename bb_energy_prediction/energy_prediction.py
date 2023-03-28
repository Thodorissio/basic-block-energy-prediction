import torch
import json
import argparse
from bb_energy_prediction import models
from bb_energy_prediction.evaluate import predict

if __name__ == "__main__":
    """CLI tool for predicting energy of basic blocks using the custom lstm model"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbs_path",
        type=str,
        help="path to file containing basic blocks. File format should have @bb_name: at the start of each basic block and each instruction should be on a new line.",
    )
    parser.add_argument(
        "--results_save_dir", nargs="?", const=1, type=str, help="path to save results"
    )

    args = parser.parse_args()

    bbs_path = args.bbs_path
    results_save_dir = args.results_save_dir

    bbs = {}
    with open(bbs_path) as fIn:
        try:
            for line in fIn:
                if line[0] == "@":
                    bb_name = line.split()[-1].rstrip(":")
                    bbs[bb_name] = []
                else:
                    bbs[bb_name].append(line.split("=")[0].rstrip())
        except:
            raise Exception("Invalid basic block file format")

    basic_blocks = list(bbs.values())

    model_dir = "../model_checkpoints/lstm_vocab_models/base_model"
    with open("../model_checkpoints/lstm_vocab_models/vocab.json") as file:
        vocab = json.load(file)
    with open(f"{model_dir}/additional_attributes.json") as json_file:
        model_config = json.load(json_file)
    model_params = model_config["model_params"]

    model = models.LSTM_Regressor(
        vocab_size=len(vocab), custom_embs=True, **model_params
    )

    model.load_state_dict(torch.load(f"{model_dir}/model"))
    model.cuda()

    results = predict(model=model, test_bbs=basic_blocks, vocab=vocab)

    res_dict = {
        bb_name: f"{str(round(energy, 3))} * (61μJ)"
        for bb_name, energy in zip(bbs.keys(), results)
    }

    if results_save_dir:
        with open(results_save_dir, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=4, ensure_ascii=False)
    else:
        print(res_dict)
