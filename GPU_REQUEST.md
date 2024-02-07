## Connect to a node with a GPU

The simplest way to connect to a node with a gpu consists in printing the list of nodes on your server with:
```
    sinfo --format="%N %.6D %P %G"
```

Then connect to a node with
```
    ssh <username>@<nodename>
```
But with this technique, you won't be able to access your directories and data.


Instead, use ``srun``. The simplest gpu request is:
```
srun -p gpu --gres=gpu:1 --pty bash
```

But you can provide more specifications:

```
srun --partition=gpu --time="4:00:00" --gres="gpu:TeslaA100:1" --cpus-per-task=32 --mem-per-gpu=16G --nodes=1 --ntasks-per-node=1 -J pbtrack --pty zsh -l
```

## Check GPU access

double check you are correctly on the node with

```
hostname
```

you should read something like ``mb-mil110.cism.ucl.ac.be`` on *Manneback* or ``drg2-w018`` on *Dragon2*. <br>
Another way to verify this consists in checking a GPU is available on your node:
```
  nvidia-smi
```

<p align="center">
<img src="../figs/nvidia_smi.png" width="600px" align="center">
</p> 