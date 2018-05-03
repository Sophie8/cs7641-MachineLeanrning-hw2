import time
from time import clock
from itertools import product
from array import array
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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
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
import java.util.Random as Random

N = range(100, 210, 10)
T=49
maxIters = 1001

values = []
weights = []
maximumValue = 0
copiesPerElement = []

PATH = "/Users/shuyi/Documents/OMSCS-master/MachineLearning/HW2/"
outfile = 'Knapsack/KS_@ALG@_LOG.txt'
ef = KnapsackEvaluationFunction(values, weights, maximumValue, copiesPerElement)
cf = SingleCrossOver()
random = Random()

# RHC
fname_rhc = PATH + outfile.replace('@ALG@','RHC')
with open(fname_rhc,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        values = []
        weights = []
        maximumValue = 0
        copiesPerElement = []
        for i in range(nn):
            values.append(random.nextDouble())
            weights.append(random.nextDouble())
            copiesPerElement.append(1)
        maximumValue = sum(weights)/2
        fill = [2] * nn
        ranges = array('i', fill) #array('i', [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        df = DiscreteDependencyTree(.1, ranges)
        ef = KnapsackEvaluationFunction(values, weights, maximumValue, copiesPerElement)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = time.clock()-start
        score = ef.value(rhc.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_rhc,'a') as f:
                f.write(st)



# SA
fname_sa= PATH + outfile.replace('@ALG@','SA')
with open(fname_sa,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        values = []
        weights = []
        maximumValue = 0
        copiesPerElement = []
        for i in range(nn):
            values.append(random.nextDouble())
            weights.append(random.nextDouble())
            copiesPerElement.append(1)
        maximumValue = sum(weights)/2
        CE = 0.75
        fill = [2] * nn
        ranges = array('i', fill) #array('i', [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        df = DiscreteDependencyTree(.1, ranges)
        ef = KnapsackEvaluationFunction(values, weights, maximumValue, copiesPerElement)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = time.clock()-start
        score = ef.value(sa.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_sa, 'a') as f:
                f.write(st)


#GA
pop = 50
mate = 10
mutate = 10
fname_ga = PATH + outfile.replace('@ALG@','GA')
with open(fname_ga,'w') as f:
    f.write('sample,fitness,time\n')
    for nn in N:
        values = []
        weights = []
        maximumValue = 0
        copiesPerElement = []
        for i in range(nn):
            values.append(random.nextDouble())
            weights.append(random.nextDouble())
            copiesPerElement.append(1)
        maximumValue = sum(weights)/2
        fill = [2] * nn
        ranges = array('i', fill) #array('i', [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        ef = KnapsackEvaluationFunction(values, weights, maximumValue, copiesPerElement)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        start = clock()
        for i in range(0,maxIters,10):

            fit.train()
        elapsed = time.clock()-start
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
        values = []
        weights = []
        maximumValue = 0
        copiesPerElement = []
        for i in range(nn):
            values.append(random.nextDouble())
            weights.append(random.nextDouble())
            copiesPerElement.append(1)
        maximumValue = sum(weights)/2
        fill = [2] * nn
        ranges = array('i', fill) #array('i', [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        ef = KnapsackEvaluationFunction(values, weights, maximumValue, copiesPerElement)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        start = clock()
        for i in range(0,maxIters,10):
            fit.train()
        elapsed = time.clock()-start
        score = ef.value(mimic.getOptimal())
        st = '{},{},{}\n'.format(nn,score,elapsed)
        print st
        with open(fname_mc,'a') as f:
                f.write(st)
