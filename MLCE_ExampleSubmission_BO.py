# Group information
group_names     = ['Name 1','Name 2']
cid_numbers     = ['000000','111111']
oral_assessment = [0, 1]

# Helper Class
class RandomSelection:
    def __init__(self, X_searchspace, objective_func, batch): 
        self.X_searchspace = X_searchspace
        self.batch         = batch

        random_searchspace = [self.X_searchspace[random.randrange(len(self.X_searchspace))] for c in range(batch)]
        self.random_Y      = objective_func(random_searchspace)


# BO class
class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func):
        start_time = datetime.timestamp(datetime.now())

        self.X_initial     = X_initial
        self.X_searchspace = X_searchspace
        self.iterations    = iterations
        self.batch         = batch

        self.Y     = objective_func(self.X_initial)
        self.time  = [datetime.timestamp(datetime.now())-start_time]
        self.time += [0]*(len(self.X_initial)-1)
        start_time = datetime.timestamp(datetime.now())
        
        for self.iteration in range(iterations):
            random_selection = RandomSelection(self.X_searchspace, objective_func, self.batch)
            self.Y           = np.concatenate([self.Y, random_selection.random_Y])
            self.time        += [datetime.timestamp(datetime.now())-start_time]
            self.time        += [0]*(len(random_selection.random_Y)-1)
            start_time = datetime.timestamp(datetime.now())

# BO Execution Block

X_initial = ([[33, 6.25, 10, 20, 20, 'celltype_1'],
              [38, 8, 20, 10, 20, 'celltype_3'],
              [37, 6.8, 0, 50, 0, 'celltype_1'],
              [36, 6.0, 20, 20, 10, 'celltype_2']])

temp = np.linspace(30, 40, 5)
pH   = np.linspace(6, 8, 5)
f1   = np.linspace(0, 50, 5)
f2   = np.linspace(0, 50, 5)
f3   = np.linspace(0, 50, 5)
celltype = ['celltype_1','celltype_2','celltype_3']

X_searchspace = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]

BO_m = BO(X_initial, X_searchspace, 15, 5, objective_func)