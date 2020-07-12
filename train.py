import argparse
import mlflow.torchscript
import torch
import transformers
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='torchscript gpt2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


tokenizer_class = transformers.GPT2Tokenizer
tokenizer = tokenizer_class.from_pretrained('gpt2')
prompt_text = 'My life has been changed'
input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")


class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = transformers.GPT2LMHeadModel.from_pretrained('gpt2', torchscript=True).eval()
        self.model = model

    def forward(self, inp, past):
        out, past = self.model(inp, past)
        return out, torch.stack(past)


def train(model):
    """
    This is supposed to be the actual training loop but for
    the sake of example, instead of training, we just return
    True
    """
    return True


with mlflow.start_run() as run:
    mlflow.log_params(vars(args))
    # Actual training loop would have come here
    model = ModelWrapper()
    single_input = input_ids[:, -1].unsqueeze(-1)
    output, past = model(single_input, past=None)
    traced = torch.jit.trace(model, (single_input, past))
    train(traced)
    print("Saving TorchScript Model ..")
    mlflow.torchscript.log_model(traced, artifact_path='model', conda_env='env.yml')
