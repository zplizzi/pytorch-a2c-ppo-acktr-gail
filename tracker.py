import datetime
import os
import time

import torch
from tensorboardX import SummaryWriter
import wandb

EXP_ROOT = "/home/zplizzi/fileserver/experiments/"
TENSORBOARD_ROOT = "/home/zplizzi/fileserver/tensorboard/"

class WandBTracker:
    def __init__(self, name=None, args=None):
        if name is not None:
            self.name = name
        else:
            self.name = "pt_baseline" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(self.name)
        print(args)
        # wandb.init(project="test", resume=True, config=args)
        wandb.init(project="test", resume=self.name)
        wandb.run.description = self.name
        # Commented this line because it was throwing errors
        # wandb.run.save()
        # TODO: log args, fix device

        # This probably should go somewhere better than here
        # But the goal is to catch SIGTERM which is unhandled by default
        # in python, and is what is sent by "docker stop" and kube analogs
        import signal
        import sys
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def checkpoint_model(self, model, i, model_name="model"):
        print(f"beginning checkpoint of {model_name}")
        # filename = f"{model_name}.{i}.pytorch"
        filename = f"{model_name}.pytorch"
        # Write model to local dir
        path = os.path.join(wandb.run.dir, filename)
        # This probably won't work not in dataparallel
        try:
            torch.save(model.module.state_dict(), path)
        except AttributeError:
            torch.save(model.state_dict(), path)
        # Send up to cloud
        wandb.save(filename)
        wandb.run.summary["checkpoint_i"] = i
        print(f"finished checkpoint of {model_name}")

    def try_continue_training(self, model, model_name="model"):
        """Returns the resumed iteration number."""
        filename = f"{model_name}.pytorch"
        # if not wandb.run.resumed:
        #     print("not resuming training")
        #     return 0
        try:
            # this would be more elegant but the summary is overwritten
            # i = wandb.run.summary["checkpoint_i"]
            api = wandb.Api()
            # TODO: set the project properly
            run = api.run(f"zplizzi/test/{self.name}")
            history = run.history(samples=10000)
            # this is actually the last step recorded, not the ckpt step
            i = max(history._step)
        except (wandb.CommError, AttributeError):
            print("attempted resume but no checkpoint recorded")
            return 0

        path = run.file(filename).download(replace=True).name

        print("resuming training!")
        # print(wandb.restore(filename))
        # path = wandb.restore(filename).name
        model.load_state_dict(torch.load(path))
        # TODO: will this work or try to overwrite history?

        # remove downloaded file
        import os
        os.remove(path)
        return i

    def log_weight_histograms(self, model, i, model_name="model"):
        pass

    def log_grad_histograms(self, model, i, model_name="model"):
        pass

    def add_histogram(self, tag, data, i):
        if type(data) == torch.Tensor:
            data = data.detach().cpu()
        wandb.log({tag: wandb.Histogram(data)}, step=i)

    def add_chart(self, tag, chart, i):
        wandb.log({tag: chart}, step=i)

    def add_scalar(self, tag, value, i):
        wandb.log({tag: value}, step=i)

    def add_image(self, tag, value, i):
        wandb.log({tag: [wandb.Image(value, caption="Label")]}, step=i)

    def watch(self, model):
        wandb.watch(model, "all")

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  #noqa
            self.last_time = time.time()
            if i % 10 == 0:
                self.add_scalar("timings/iterations-per-sec", 1/dt, i)
                self.add_scalar("timings/samples-per-sec", batch_size/dt, i)
        except AttributeError:
            self.last_time = time.time()

class TensorboardTracker:
    def __init__(self, name=None, args=None):
        if name is not None:
            self.name = name
        else:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(self.name)

        self.writer = SummaryWriter(f"{TENSORBOARD_ROOT}{self.name}")

    def checkpoint_model(self, model, i, model_name="model"):
        assert "." not in model_name
        folder = f"{EXP_ROOT}{self.name}/ckp/"
        filename = f"{model_name}.{i}.pt"
        os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), folder + filename)

    def get_checkpoint_filename(self, model_name="model"):
        try:
            files = os.listdir(f"{EXP_ROOT}{self.name}/ckp/")
        except FileNotFoundError:
            return None
        matching_files = [x for x in files if x.split(".")[0] == model_name]
        last_ckpt = max(matching_files, key=lambda x: int(x.split(".")[1]))
        return last_ckpt

    def load_checkpoint_file(self, model, filename):
        """Load a saved checkpoint file into `model`."""
        filepath = f"{EXP_ROOT}{self.name}/ckp/{filename}"
        model.load_state_dict(torch.load(filepath))

    def try_continue_training(self, model, model_name="model"):
        filename = self.get_checkpoint_filename(model_name)
        if filename is None:
            return 0
        self.load_checkpoint_file(model, filename)
        # return the iteration number
        i = int(filename.split(".")[1])
        print(f"loading checkpoint from iteration {i}")
        return i

    def log_weight_histograms(self, model, i, model_name="model"):
        """Log histograms of each model parameter."""
        for name, val in model.named_parameters():
            self.writer.add_histogram(f"{model_name}_{name}", val, i)

    def log_grad_histograms(self, model, i, model_name="model"):
        """Log histograms of the gradient of each model parameter."""
        for name, val in model.named_parameters():
            self.writer.add_histogram(f"{model_name}_grad_{name}", val.grad, i)

    def add_scalar(self, tag, value, i):
        self.writer.add_scalar(tag, value, i)

    def add_image(self, tag, value, i):
        self.writer.add_image(tag, value, i)

    def log_args(self, args):
        print(args)
