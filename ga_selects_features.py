import sys
import random
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

class GASelectsFeatures():
    def __init__(self,
                 input_model: List[str],
                 n_population: int=100,
                 n_gen: int=50,
                 cxpb=0.5,
                 mutpb=0.8):
        self.input_model = input_model
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
        pop = self.toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(1)

        # pop: Population defined earlier
        # toolbox: toolbox containing all the operator defined
        # cxpb: The probability of mating two individuals.
        # mutpb: The probability of mutating an individual. We are keeping it high to show the impact
        # ngen: The number of generation.
        pop, log = algorithms.eaSimple(pop,
                                       self.toolbox,
                                       cxpb=self.cxpb,
                                       mutpb=self.mutpb,
                                       ngen=self.n_gen,
                                       halloffame=hof,
                                       stats=self.stats,
                                       verbose=True)
        # Get the best individual
        best = hof.items[0]
        input_model_select = []
        for i in range(0, len(best)):
            if best[i] == 1:
                input_model_select.append(self.input_model[i])
        return input_model_select

    def __arquicteture_ga(self,
                          model: LGBMRegressor,
                          X_train: pd.DataFrame,
                          X_val: pd.DataFrame,
                          y_train: pd.Series,
                          y_val: pd.Series):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        creator.create("Individual", list, fitness=creator.FitnessMin)

        ind_size = len(self.input_model)
        self.toolbox = base.Toolbox()

        self.toolbox.register("attrib_bin", random.randint, 0, 1)

        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attrib_bin, n=ind_size)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)


        # Two points crossover
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Bit flip mutation The indpb argument is the probability of each attribute to be flipped
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)

        # Select the best individual among tournsize randomly chosen individuals
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Register the fitness function defined above in the toolbox
        self.toolbox.register("evaluate",
                              self.__evaluate,
                              model=model,
                              input_model=self.input_model,
                              X_train=X_train,
                              y_train=y_train,
                              X_val=X_val,
                              y_val=y_val)

        # Define the statistics to be shown during the algorithm run.
        # We have selected minimum, maximum and average accuracy for each generation of run
        # Decision will, however, be taken based on maximum accuracy as defined earlier
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("Mean", np.mean, axis=0)
        self.stats.register("Max", np.max, axis=0)
        self.stats.register("Min", np.min, axis=0)

    def __evaluate(self,
                   individual: List[int],
                   model: LGBMRegressor,
                   input_model: List[str],
                   X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   y_val: pd.Series) -> Tuple[float, None]:
        sum_features = np.sum(individual)
        if sum_features == 0:
            return sys.float_info.max,
        else:
            input_model_select = []
            for k in range(0, len(individual)):
                if individual[k] == 1:
                    input_model_select.append(input_model[k])
            X_train = X_train[input_model_select]
            model.fit(X_train, y_train)

            X_val_select = X_val[input_model_select]
            y_val_pred = model.predict(X_val_select)

            rmse = self.root_mean_squared_error(y_val, y_val_pred)
            return rmse,

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
