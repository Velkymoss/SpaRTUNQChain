import logging
import os
import random
from typing import Any

import numpy as np
import torch
from domiknows.program.model.base import Mode

from tempQchain.logger import get_logger
from tempQchain.programs.program_fr import program_declaration_tb_dense_fr
from tempQchain.readers.temporal_reader import TemporalReader
from tempQchain.utils import get_class_distribution

logger = get_logger(__name__)

logging.basicConfig(level=logging.INFO)


def main(args: Any) -> None:
    SEED = args.seed
    logger.info(f"Model: {args.model}")
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Constraint analysis - loading constraints test dataset...")

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else "cpu"

    test_constraints_file = "tb_dense_test_constraints.json"
    test_constraints_path = os.path.join(args.data_path, test_constraints_file)

    test_constraints_set = TemporalReader.from_file(
        file_path=test_constraints_path, question_type="FR", batch_size=args.batch_size
    )

    class_distribution = get_class_distribution(test_constraints_set)
    logger.info(f"Test constraints class distribution: {class_distribution}")

    program = program_declaration_tb_dense_fr(
        cur_device,
        pmd=args.pmd,
        beta=args.beta,
        sampling=args.sampling,
        sampleSize=args.sampling_size,
        dropout=args.dropout,
        constraints=args.constraints,
        class_weights=None,
    )

    pretrain_model = torch.load(
        os.path.join("models", args.model),
        map_location={
            "cuda:0": cur_device,
            "cuda:1": cur_device,
            "cuda:2": cur_device,
            "cuda:3": cur_device,
            "cuda:4": cur_device,
            "cuda:5": cur_device,
        },
    )
    pretrain_dict = pretrain_model
    current_dict = program.model.state_dict()

    new_state_dict = {k: v if k not in pretrain_dict else pretrain_dict[k] for k, v in current_dict.items()}
    program.model.load_state_dict(new_state_dict)

    logger.info("Program instance created for constraint analysis.")

    program.model.to(cur_device)
    program.model.mode(Mode.TEST)
    program.model.reset()

    logger.info("Verifying results for the test constraints set...")
    program.verifyResultsLC(test_constraints_set, device=cur_device)
    logger.info("Constraint analysis completed.")

    # all_chains = []
    # for batch_idx, datanode in enumerate(program.populate(test_constraints_set, device=cur_device)):
    #     id_map = {}
    #     for pos, q in enumerate(datanode.getChildDataNodes()):
    #         qid_raw = q.getAttribute("id")
    #         try:
    #             qid_val = int(qid_raw.item())
    #         except Exception:
    #             try:
    #                 qid_val = int(qid_raw)
    #             except Exception:
    #                 qid_val = pos

    #         try:
    #             label_val = int(q.getAttribute(answer_class, "label"))
    #         except Exception:
    #             label_val = None
    #         pred_val = None
    #         try:
    #             logits = q.getAttribute(answer_class, "local/argmax")
    #             pred_val = int(torch.argmax(logits))
    #         except Exception:
    #             try:
    #                 soft = q.getAttribute(answer_class, "local/softmax")
    #                 pred_val = int(torch.argmax(soft))
    #             except Exception:
    #                 pred_val = None

    #         id_map[qid_val] = {
    #             "pos": pos,
    #             "qid": qid_val,
    #             "question": q.getAttribute("question"),
    #             "story": q.getAttribute("story"),
    #             "label": label_val,
    #             "prediction": pred_val,
    #         }

    #     chains_in_batch = []
    #     for q in datanode.getChildDataNodes():
    #         qid_raw = q.getAttribute("id")
    #         try:
    #             qid = int(qid_raw.item())
    #         except Exception:
    #             try:
    #                 qid = int(qid_raw)
    #             except Exception:
    #                 continue

    #         rel_raw = q.getAttribute("relation") or ""
    #         rel_str = rel_raw if isinstance(rel_raw, str) else str(rel_raw)
    #         if not rel_str:
    #             continue
    #         parts = rel_str.split(",")
    #         constraint = parts[0]
    #         if constraint not in ("transitive", "symmetric"):
    #             continue

    #         related_ids = [int(p) for p in parts[1:] if p.strip()]

    #         primary = id_map.get(qid, None)
    #         related = [
    #             id_map.get(
    #                 rid, {"qid": rid, "pos": None, "question": None, "story": None, "label": None, "prediction": None}
    #             )
    #             for rid in related_ids
    #         ]

    #         chain = {
    #             "batch_idx": batch_idx,
    #             "constraint": constraint,
    #             "primary": primary,
    #             "related": related,
    #         }
    #         chains_in_batch.append(chain)

    #     if chains_in_batch:
    #         all_chains.extend(chains_in_batch)

    # output_file = getattr(args, "output_file", os.path.join(args.data_path, "final_chain_questions.json"))
    # output_dir = os.path.dirname(output_file)
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(all_chains, f, indent=4)
