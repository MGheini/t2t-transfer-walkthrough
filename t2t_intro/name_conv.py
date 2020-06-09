import sys
problem_name = sys.argv[1]

problem_name_lc = problem_name.lower()
label = ""
for x in range(len(problem_name)):
        if x != 0 and problem_name[x] != problem_name_lc[x]:
                label += "_"
        label += problem_name_lc[x]

print(label)
