import subprocess
from tqdm import tqdm

with open("starcoder_pretrained.txt") as f:
    text = ''.join(f.readlines())
f.close()
tests = text.split("Task_id: ")[1:]

scores = []
for test in tqdm(tests):
    assertions = []
    next_test = test.find('assert')
    while next_test != -1:
        test = test[next_test:].lstrip()
        assertions += [test[:test.find('\n')]]
        test = test[test.find('\n'):]
        next_test = test[1:].find('assert')

    if len(assertions) <= 1:
        scores += [0]
        continue

    for i in range(len(assertions)):
        with open("file_" + str(i) + ".py", 'w') as f:
            f.write(assertions[i])
        f.close()

    avg_code_bleu = 0
    comparisons = 0
    for i in range(len(assertions)):
        for j in range(i+1, len(assertions)):
            output = subprocess.run(['python', 'calc_code_bleu.py', '--refs', 'file_'+ str(i) + '.py', '--hyp', 'file_'+ str(j) + '.py', '--lang', 'python']
                                    , capture_output=True).stdout.decode()
            avg_code_bleu += float(output[output.find("CodeBLEU score:") + len("CodeBLEU score:"):].strip())
            comparisons += 1
    scores += [avg_code_bleu / comparisons]

with open("bleu_scores_codellama_pretrained.csv", 'w') as f:
    f.write("Task Id,CodeBLEU\n")
    for i in range(len(scores)):
        f.write(str(i) + "," + str(scores[i]) + "\n")
    f.close()
