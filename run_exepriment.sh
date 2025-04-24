CUDA_VISIBLE_DEVICES=7 python3 task2x_combined_regression.py
CUDA_VISIBLE_DEVICES=5,6,7 python3 new_test_eval.py --model_type regression --model_path models/best_single_head_regression_model_epoch_4.pt --rounding floor
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ensemble_test_eval.py --model_path models/ --ensemble_info best_ensemble_info.pkl