import sys
import random
from operator import attrgetter
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

class Chromosome(object):
    """Implements a chromosome container.

    Chromosome represents the list of genes, whereas each gene is
    the name of feature. Creating the chromosome, we generate
    the random sample of features.

    Args:
        genes:
            List of all feature names.
        size:
            Number of genes in chromosome,
            i.e. number of features in the model.
    """

    def __init__(self, genes, size):
        self.genes = self.generate(genes, size)

    def __repr__(self):
        return ' '.join(self.genes)

    def __get__(self, instance, owner):
        return self.genes

    def __set__(self, instance, value):
        self.genes = value

    def __getitem__(self, item):
        return self.genes[item]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __len__(self):
        return len(self.genes)

    @staticmethod
    def generate(genes, size):
        return random.sample(genes, size)

class GASelectsKFeatures():
    def __init__(self,
                 input_model: List[str],
                 n_features: int,
                 n_population: int=100,
                 n_gen: int=25,
                 cxpb=0.2,
                 mutpb=0.8):
        self.input_model = input_model
        self.n_features = n_features
        self.n_population = n_population
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb

        self.toolbox = None
        self.stats = None

    def transform(self,
                  model: LGBMRegressor,
                  X_train: pd.DataFrame,
                  X_val: pd.DataFrame,
                  y_train: pd.Series,
                  y_val: pd.Series) -> List[str]:

        self.__arquicteture_ga(model,
                               X_train,
                               X_val,
                               y_train,
                               y_val)
          # raise population
        pop = self.toolbox.population(self.n_population)

        # keep track of the best individuals
        hof = tools.HallOfFame(1)

        # pop: Population defined earlier
        # toolbox: toolbox containing all the operator defined
        # cxpb: The probability of mating two individuals.
        # mutpb: The probability of mutating an individual. We are keeping it high to show the impact
        # ngen: The number of generation.
        algorithms.eaMuPlusLambda(pop,
                                  self.toolbox,
                                  mu=10,
                                  lambda_=30,
                                  cxpb=self.cxpb,
                                  mutpb=self.mutpb,
                                  ngen=self.n_gen,
                                  stats=self.stats,
                                  halloffame=hof,
                                  verbose=True)

        return hof[0].genes

    def __arquicteture_ga(self,
                          model: LGBMRegressor,
                          X_train: pd.DataFrame,
                          X_val: pd.DataFrame,
                          y_train: pd.Series,
                          y_val: pd.Series):
        creator.create('FitnessMin', base.Fitness, weights=(-1,))
        creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

        # register callbacks
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            'individual', self.__init_individual, creator.Individual,
            genes=self.input_model, size=self.n_features)
        self.toolbox.register(
            'population', tools.initRepeat, list, self.toolbox.individual)

      

        # register fitness evaluator
        self.toolbox.register("evaluate",
                         self.__evaluate,
                         model=model,
                         X_train=X_train,
                         y_train=y_train,
                         X_val=X_val,
                         y_val=y_val)
        # register standard crossover
        self.toolbox.register('mate', tools.cxTwoPoint)
        # replace mutation operator by our custom method
        self.toolbox.register('mutate', self.__mutate, genes=self.input_model, pb=0.4)
        # register elitism operator
        self.toolbox.register('select', self.__select_best)

        # setting the statistics (displayed for each generation)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('min', np.min)
        stats.register('max', np.max)

    def __init_individual(self,
                          ind_class,
                          genes=None,
                          size=None):
        return ind_class(genes, size)

    def __evaluate(self,
                   individual: List[int],
                   model: LGBMRegressor,
                   X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   y_val: pd.Series) -> Tuple[float, None]:
        
        X_train = X_train[individual.genes]
        model.fit(X_train, y_train)

        X_val = X_val[individual.genes]
        y_val_pred = model.predict(X_val)

        rmse = self.root_mean_squared_error(y_val, y_val_pred)
        return rmse,

    def __mutate(self,
                 individual,
                 genes=None, pb=0):
        """Custom mutation operator used instead of standard tools.

        We define the maximal number of genes, which can be mutated,
        then generate a random number of mutated genes (from 1 to max),
        and implement a mutation.

        Args:
            individual:
                The list of features (genes).
            genes:
                The list of all features.
            pb:
                Mutation parameter, 0 < pb < 1.

        Returns:
             Mutated individual (tuple).
        """

        # set the maximal amount of mutated genes
        n_mutated_max = max(1, int(len(individual) * pb))

        # generate the random amount of mutated genes
        n_mutated = random.randint(1, n_mutated_max)

        # pick up random genes which need to be mutated
        mutated_indexes = random.sample(
            [index for index in range(len(individual.genes))], n_mutated)

        # mutation
        for index in mutated_indexes:
            individual[index] = random.choice(genes)

        return individual,

    def __select_best(self,
                      individuals,
                      k,
                      fit_attr='fitness'):
        """Custom selection operator.

        The only difference with standard 'selBest' method
        (select k best individuals) is that this method doesn't select
        two individuals with equal fitness value.

        It is done to prevent populations with many duplicate individuals.
        """

        return sorted(
            set(individuals),
            key=attrgetter(fit_attr),
            reverse=True)[:k]

    def root_mean_squared_error(self,
                                y_true: pd.Series,
                                y_pred: pd.Series) -> float:
        """Faz o cálculo do RMSE.

        Parameters
        ----------
        y_true : pd.Series
            Valor real.
        y_pred : pd.Series
            Valor predito pelo modelo.

        Returns
        -------
        float
            Valor do RMSE.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
