import sys, os
import time

def main():

        template = ""
        with open("template") as t:
                template = t.read().split("!HP_SPLIT")
        problems = {}
        problem_path = "/home/nlg-05/gheini/miniconda2/envs/t2tenv/lib/python3.5/site-packages/tensor2tensor/data_generators/"
        batch_size = "4096"
        vocab_size = "32768"
        save_problem = None
        res_tokens = []
        hparams=None
        pbs = os.listdir(problem_path)
        for i in range(len(pbs)):
                pbs[i] = pbs[i].split('.')[0]
        problems = set(pbs)
        args = sys.argv[1:]
        timestamp = str(int(time.time()))
        if len(args) % 2 == 1:
                print("invalid arguments passed")
                exit()
        argdict = {}
        for i in range(len(args)//2):
                argdict[args[i*2]] = args[i*2+1]
        if argdict.get("--batch_size") is not None:
                batch_size = argdict["--batch_size"]
        if argdict.get("--vocab_size") is not None:
                vocab_size = argdict["--vocab_size"]
        if argdict.get("--save_problem") is not None:
                save_problem = argdict["--save_problem"]
        if argdict.get("--hparams") is not None:
                hparams = argdict["--hparams"]
        if argdict.get("--problem_path") is not None:
                problem_path = argdict["--problem_path"]
        if argdict.get("--res_tokens") is not None:
                res_tokens = argdict["--res_tokens"].split(",")
                print("Additional reserved tokens are: {}".format(res_tokens))
        if save_problem is not None:
                if save_problem in problems:
                        save_problem = save_problem# + timestamp
        else:
                save_problem="TempProblem" + timestamp

        with open(problem_path + save_problem + ".py", "w") as outfile:
                first_half = template[0].replace("!VOCAB_SIZE", vocab_size).replace("!ProblemName", save_problem).replace("!RES_TOKENS", str(res_tokens))
                outfile.write(first_half)
                save_problem_lc = save_problem.lower()
                label = ""
                for x in range(len(save_problem)):
                        if x != 0 and save_problem[x] != save_problem_lc[x]:
                                label += "_"
                        label += save_problem_lc[x]
                if batch_size != "4096" or hparams is not None:
                        second_half = template[1].replace("!HPARAMS_NAME", "hparams" + timestamp)
                        hparams_assign = "hparams.batch_size = " + batch_size
                        if hparams is not None:
                                for hp in hparams.split(","):
                                        hparams_assign += "\n        hparams." + hp.replace("=", " = ")
                        second_half = second_half.replace("!HPARAMS_ASSIGN", hparams_assign)
                        outfile.write(second_half)
                        label += " " +  "hparams" + timestamp

        all_problems_contents = ""
        with open(problem_path + "all_problems.py", "r") as apfile:
                all_problems_contents = apfile.read()

        with open("ap_backup", "w+") as bp, open(problem_path + "all_problems.py", "w") as apw:
                bp.write(all_problems_contents)
                lines = all_problems_contents.splitlines()
                for line in lines:
                        if line.strip() == "]":
                                apw.write("    \"tensor2tensor.data_generators." + save_problem + "\",\n")
                        apw.write(line + "\n")

        print(label)


if __name__ == "__main__":
        main()


