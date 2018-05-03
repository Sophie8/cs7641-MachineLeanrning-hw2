
from array import array
import java.util.Random as Random
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays
from time import clock
"""
Commandline parameter(s):
   none
"""

N=range(100, 210, 10)
T=49
maxIters = 1001
random = Random()
outfile = 'TSP/FOURPEAKS_@ALG@_LOG.txt'
PATH = ""
PATH = "/Users/shuyi/Documents/OMSCS-master/MachineLearning/HW2/"
points = []
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile = 'TSP/TSP_@ALG@_LOG.txt'
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N[0])
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# RHC
fname_rhc = PATH + outfile.replace('@ALG@','RHC')
with open(fname_rhc,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        fill = [nn] * nn
        ranges = array('i', fill)
        points = [[0 for x in xrange(2)] for x in xrange(nn)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()
        odd = DiscreteUniformDistribution(ranges)
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = clock()-start
        score = ef.value(rhc.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_rhc,'a') as f:
            f.write(st)



# SA
CE = 0.15
fname_sa = PATH + outfile.replace('@ALG@','SA')
with open(fname_sa,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        fill = [nn] * nn
        ranges = array('i', fill)
        points = [[0 for x in xrange(2)] for x in xrange(nn)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        odd = DiscreteUniformDistribution(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = clock()-start
        score = ef.value(sa.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_sa,'a') as f:
            f.write(st)

#GA
pop = 100
mate = 50
mutate = 10
fname_ga = PATH + outfile.replace('@ALG@','GA')
with open(fname_ga,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        fill = [nn] * nn
        ranges = array('i', fill)
        points = [[0 for x in xrange(2)] for x in xrange(nn)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()
        odd = DiscreteUniformDistribution(ranges)
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        cf = TravelingSalesmanCrossOver(ef)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = clock()-start
        score = ef.value(ga.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_ga,'a') as f:
            f.write(st)


#MIMIC
samples = 100
keep = 50
m = 0.7
fname_mc = PATH + outfile.replace('@ALG@','MIMIC')

with open(fname_mc,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        points = [[0 for x in xrange(2)] for x in xrange(nn)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()
        fill = [nn] * nn
        ranges = array('i', fill)
        df = DiscreteDependencyTree(m, ranges)
        odd = DiscreteUniformDistribution(ranges)
        ef = TravelingSalesmanSortEvaluationFunction(points)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = clock()-start
        score = ef.value(mimic.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_mc,'a') as f:
            f.write(st)



