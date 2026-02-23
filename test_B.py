import os, shutil, argparse
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml

from lmdb_data_loader_A import LmdbDataset

from models.my_model4 import ResnetConformer_seddoa_nopool_2023

from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file

from utils.sed_doa import process_foa_input_sed_doa_labels, process_raw_mic_input, SedDoaLoss, SedDoaResult_2023
# ---------------------------------------------------------------------
# Evaluate on unseen test data
# ---------------------------------------------------------------------
from ast import main
import sys

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def main(args):
    print('Load best model weights')
    log_output_folder = os.path.dirname(args['test']['test_output_dir'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['test']['test_output_dir'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)


    data_process_fn = process_raw_mic_input
    result_class = SedDoaResult_2023
    criterion = SedDoaLoss(loss_weight=[0.1,1])
    model = ResnetConformer_seddoa_nopool_2023(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])

    test_split = [4]
    test_dataset = LmdbDataset(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=None, data_process_fn=data_process_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    #pdb.set_trace()
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model'], map_location=device))
    logger.info(model)

    start_time = time.time()
    model.eval()
    test_loss = []
    test_result = result_class(segment_length=args['data']['segment_len'])
    #pdb.set_trace()
    for data in test_dataloader:
        input = data['input'].to(device)
        target = data['target'].to(device)
        with torch.no_grad():
            output = model(input) # [1, 100, 52]

            min_len = min(output.shape[1], target.shape[1]) # 4
            output = output[:, :min_len, :]
            target = target[:, :min_len, :]
            loss = criterion(output, target)
            test_loss.append(loss.item())
        test_result.add_items(data['wav_names'], output)
    output_dict = test_result.get_result()
    test_time = time.time() - start_time
    
    dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'best_results')
    # if os.path.exists(dcase_output_val_dir):
    #     shutil.rmtree(dcase_output_val_dir)
    os.makedirs(dcase_output_val_dir, exist_ok=True)
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
        write_output_format_file(output_file, perfile_out_dict)
    
    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    
    # 주의: get_SELD_Results가 반환하는 값의 개수가 2024 버전과 다를 수 있음.
    # 일반적인 2023 포맷에 맞게 수정 (실행 시 에러가 나면 반환 변수 개수 확인 필요)
    # 보통 2023은: ER, F, LE, LR, SELD_score, class_wise 순서임 (dist_err 관련 제외)
    results = score_obj.get_SELD_Results(dcase_output_val_dir)
    
    # 반환값 unpacking (유동적으로 처리)
    val_ER = results[0]
    val_F = results[1]
    val_LE = results[2]
    val_LR = results[3]
    val_seld_scr = results[4]
    classwise_test_scr = results[-1] # 마지막이 classwise

    logger.info('ER/F/LE/LR/SELD: {:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr))
    print('ER/F/LE/LR/SELD: {:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr))
    
    # [수정] Classwise 결과에 ER 추가
    print('Classwise results on unseen test data')
    # 헤더에 ER 추가
    print('Class\tER\tF\tAE\tLR\tSELD_score') 
    
    for cls_cnt in range(0, 13):
        # classwise_test_scr[0]이 ER에 해당합니다.
        print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
            cls_cnt,
            classwise_test_scr[0][cls_cnt], # [추가됨] ER (Error Rate)
            classwise_test_scr[1][cls_cnt], # F-score
            classwise_test_scr[2][cls_cnt], # AE (Localization Error)
            classwise_test_scr[3][cls_cnt], # LR (Localization Recall)
            classwise_test_scr[4][cls_cnt]  # SELD Score
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='config_2023', help='name of config')
    input_args = parser.parse_args()

    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)