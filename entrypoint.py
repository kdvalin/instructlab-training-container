from instructlab.training.main_ds import run_training
from instructlab.training.config import TorchrunArgs, TrainingArgs, DeepSpeedOptions
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#Required
parser.add_argument("--model", default=argparse.SUPPRESS, required=True, type=str, help="Path to the model used for training")
parser.add_argument("--data-path", default=argparse.SUPPRESS, required=True, type=str, help="Path to training data")

#Optional/provided defaults
parser.add_argument("--nnodes", default=1, type=int, help="Number of nodes used for training")
parser.add_argument("--nproc-per-node", default=1, type=int, help="Number of GPUs per node")
parser.add_argument("--ckpt-output-dir", type=str, default="/tmp", help="Path to output model checkpoints")
parser.add_argument("--num-epochs", type=int, default=1, help="Number of iterations over the training dataset")
parser.add_argument("--dolomite", action='store_true', help="Set if the model is in dolomite format, by enabling `is_padding_free`")
parser.add_argument("--effective-batch-size", type=int, default=64, help="Batch size scaling factor")
parser.add_argument("--max-batch-len", type=int, default=60000, help="Maximum length of a batch")
parser.add_argument("--data-output-dir", type=str, default="/dev/shm", help="Directory to output preprocessed data")

args = parser.parse_args()

run_training(
        torch_args=TorchrunArgs(
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=0,
            rdzv_id=123,
            rdzv_endpoint='0.0.0.0:8888'
        ),
        train_args=TrainingArgs(
                model_path = args.model,
                data_path = args.data_path,
                ckpt_output_dir = args.ckpt_output_dir,
                data_output_dir = args.data_output_dir,
                max_seq_len = 4096,
                max_batch_len = args.max_batch_len,
                num_epochs = args.num_epochs,
                effective_batch_size = args.effective_batch_size,
                save_samples = 74999,
                learning_rate = 2e-5,
                warmup_steps = 385,
                is_padding_free = args.dolomite,
                random_seed = 42,
        ),
)
