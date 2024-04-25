# Generatlized CodeT implementation based on this paper: https://arxiv.org/abs/2207.10397
# Assumes that programs and tests are executable when concatenated (concat(x, y) runs for each x in X and y in Y)

import re
import random

class CodeT:
    def __init__(self, X, Y, n, k):
        """
        X: (List[string]) list of sampled programs
        Y: (List[string]) list of sampled test cases
        k: top k of these generated programs (k <= n)
        """

        try:
            self.programs = self.dual_execution_agreement(X, Y, n, k)
            print("Ran CodeT")
        except:
            self.programs = X[:k]
            print("No CodeT")
        

    def extract_function_name(self, code_string):
        '''
        code_string: (string) String that contains code for a function
        returns: the name of the function
        '''
        pattern = r'def\s+(\w+)\s*\('
        match = re.search(pattern, code_string)
        if match:
            return match.group(1)  # The captured function name
        else:
            return None

    def evaluate_function(self, func_string, *args, **kwargs):
        '''
        func_string: a string that contains code for a function
        *args/*kwargs: contains the arguments to the function
        returns: the function evaluated on the arguments passed in
        '''
        namespace = {}
        
        exec(func_string, globals(), namespace)

        func_name = self.extract_function_name(func_string)
        func = namespace.get(func_name)
        
        res = None
        if func is not None and callable(func):
            try:
                res = func(*args, **kwargs)
            except:
                res = None
        
        return res

    def valid_program_testcase_pair(self, x, y):
        '''
        x: a sample program hypothetical inlier
        y: a sample test case hypothetical inlier
        returns: True if the sample program passed the test case, False if not
        '''
        
        executable = x + "\n" + y
        
        try: 
            exec(executable)
            return True
        except:
            return False


    def build_consensus_set(self, X, Y, x):
        '''
        X: (List[String]) set of programs
        Y: (List[tuple]) set of test cases
        x: a sample program hypothetical inlier
        returns: (S, score) where S is the consensus set of the program x 
                and score is the score given to that consensus set
        '''
        
        S_y = []
        for y in Y:
            if self.valid_program_testcase_pair(x, y):
                S_y.append(y)
        
        S_x = []
        for program in X:
            matching_program = True
            for y in S_y:
                if (not self.valid_program_testcase_pair(program, y)):
                    matching_program = False
                    break
            if matching_program:
                S_x.append(program)
        
        S = []
        for program in S_x:
            for test in S_y:
                S.append((program, test))
        
        score = len(S)
        return S, score
    
    def dual_execution_agreement(self, X, Y, n, k):
        '''
        X: (List[string]) list of programs
        Y: (List[tuple]) test cases
        k: (int) the number of times we repeat the selection process
        returns: (S, score) where S is the consensus set with the highest 
                score based on random sampling over k iterations
        '''
        
        patience = 2 * n
        D = [(x, y) for y in Y for x in X]
        ctr = 0
        num_iters = 0
        clusters = []

        while ctr < n and num_iters < patience:
            idx = random.randint(0, len(D) - 1)
            x, y = D[idx]
            if self.valid_program_testcase_pair(x, y):
                S_x, S_y = self.get_groups(X, Y, x)
                score = len(S_x) * len(S_y)
                clusters.append((S_x, score))
                ctr += 1
            num_iters += 1
        
        if len(clusters) < 1:
            raise Exception("No matching programs and test \
                            cases were found with given patience.")
        
        clusters.sort(key = lambda x : x[1], reverse = True)
    
        ctr = 0
        n_clusters = len(clusters)
        res = []
        while ctr < k:
            cluster_idx = ctr % n_clusters
            S_x = clusters[cluster_idx][0]
            program_idx = random.randint(0, len(S_x) - 1)
            res.append(S_x[program_idx])
            ctr += 1
        
        return res

    
    def get_groups(self, X, Y, x):
        '''
        X: (List[String]) set of programs
        Y: (List[tuple]) set of test cases
        x: a sample program hypothetical inlier
        returns: (S, score) where S is the consensus set of the program x 
                and score is the score given to that consensus set
        '''
        
        S_y = []
        for y in Y:
            if self.valid_program_testcase_pair(x, y):
                S_y.append(y)
        
        S_x = []
        for program in X:
            matching_program = True
            for y in S_y:
                if (not self.valid_program_testcase_pair(program, y)):
                    matching_program = False
                    break
            if matching_program:
                S_x.append(program)
        
        return S_x, S_y
    
    def dual_execution_agreement_small(self, X, Y, k):
        '''
        X: (List[string]) list of programs
        Y: (List[tuple]) test cases
        k: (int) the number of programs we want from CodeT
        returns: List[String] of k programs
        '''
        
        test_sets = []

        for x in X:
            S_y = []
            for y in Y:
                if self.valid_program_testcase_pair(x, y):
                    S_y.append(y)
            
            test_sets.append(S_y)
        
        
        frozen_test_sets = [frozenset(S_y) for S_y in test_sets]
        test_program_map = {}

        for i in range(len(frozen_test_sets)):
            S_y  = frozen_test_sets[i]
            x = X[i]
            if S_y in test_program_map:
                test_program_map[S_y].append(x)
            else:
                test_program_map[S_y] = [x]

        clusters = []
        for S_y in test_program_map:
            S_x = test_program_map[S_y]
            score = len(S_x) * len(S_y)
            clusters.append((S_x, score))
        
        clusters.sort(key = lambda x : x[1], reverse = True)

        ctr = 0
        n_clusters = len(clusters)
        res = []
        while ctr < k:
            cluster_idx = ctr % n_clusters
            S_x = clusters[cluster_idx][0]
            program_idx = random.randint(0, len(S_x) - 1)
            res.append(S_x[program_idx])
            ctr += 1
        
        return res


if __name__ == '__main__':
    
    
    x1 = """def num_square(a):\n    return a**2"""
    x2 = """def num_square(a):\n    return a*a"""
    x3 = """def num_square(a):\n    return a*2"""
    x4 = """def num_square(a):\n    return a"""
    
    t1 = "assert num_square(1) == 1"
    t2 = "assert num_square(2) == 4"
    t3 = "assert num_square(0) == 0"
    t4 = "assert num_square(1) == 2"
    t5 = "assert num_square(3) == 6"

    X = [x1, x2, x3, x4]
    Y = [t1, t2, t3, t4, t5]
    k = 7
    n = 4
    
    codet = CodeT(X, Y, k)

    print(codet.programs)

   
