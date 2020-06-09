# t2t-transfer-walkthrough
Instructions for transfer learning using my T2T setup

**Credit:** The initial setup for running training sessions using Tensor2Tensor was provided to me by Jibia Shen. I adapted it for my own transfer learning purposes.

**Disclaimer:** I'm in no way suggesting Tensor2Tensor is the only way or necessarily the best way to train MT models. In the good old days (Winter 2019 :D), when I was experimenting with universal transfer learning I started with it and happened to train my favorite parent model so far using it. So here we are. 

**Prerequisites:** Have [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) installed, and be reasonably comfortable with it. For me, its code gets complicated very quickly. So I've always been loosely comfortable with it. You should be good by going through README.

## Running the Setup
1. Copy the files in `t2t_intro` into the directory you plan to run your experiments in.

2. Update [`problem_path` in `problem_gen.py`](https://github.com/MGheini/t2t-transfer-walkthrough/blob/master/t2t_intro/problem_gen.py#L10,L12).

3. Run an example training session using the following command to make sure you've got everything right up to this point.

    ```
    bash tfmexp.sh -w <DESIRED_EXPERIMENT_FOLDER_NAME>
                   -d <PATH_TO_DATA_DIRECTORY>
                   -l <SRD_LANG>
                   -n <DATA_SPLIT>
                   -N -p <NEW_PROBLEM_NAME>
    ```
  
    Where,

     - `<DESIRED_EXPERIMENT_FOLDER_NAME>` is the path to your experiment directory.
     - `<PATH_TO_DATA_DIRECTORY>` is the path to the directory you have your training data in. The naming of the training files should follow a specific convention that we touch on soon.
     - `<SRD_LANG>` is the source language.
     - `<DATA_SPLIT>` is the split of data you will be using. If you will be using all of the data, this will be 1. If you will be using a quarter of the data, this will be 4.
     - `<NEW_PROBLEM_NAME>` is the problem name you will be registering with Tensor2Tensor. Names should be in **UpperCamelCase**. If you are creating a problem for the first time use `-N -p`. If you are using a problem from earlier just use  `-p`.
     
I'm putting few example commands here, which is only accessible if you're signed in using your ISI account.

## Contact
Feel free to contact me on Slack or through email if you face any problems. You can simply make an issue here too.
