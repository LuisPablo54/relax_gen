from .gen import GEN
from .algorithms.alg_binary import cl_alg_stn_bin
from .algorithms.alg_quantum import cl_alg_quantum
from .algorithms.alg_EDA import cl_alg_eda
from .algorithms.alg_genetic_programming import cl_alg_gp
from .algorithms.alg_differential_evolution import cl_alg_de

class RelaxGEN(GEN):
    def __init__(self, funtion=None, population=None, **kwargs):
        super().__init__(funtion, population, **kwargs)
        self.num_genes = kwargs.get("num_genes")  
        self.num_cycles = kwargs.get("num_cycles") 
        self.selection_percent = kwargs.get("selection_percent") 
        self.crossing = kwargs.get("crossing") 
        self.mutation_percent = kwargs.get("mutation_percent") 
        self.i_min = kwargs.get("i_min")
        self.i_max = kwargs.get("i_max")
        self.optimum = kwargs.get("optimum")
        self.num_qubits = kwargs.get("num_qubits")
        self.select_mode = kwargs.get("select_mode")
        self.num_variables = kwargs.get("num_variables")
        self.data = kwargs.get("data")
        self.possibility_selection = kwargs.get("possibility_selection")
        self.metric = kwargs.get("metric")
        self.model = kwargs.get("model")
        self.max_depth = kwargs.get("max_depth")
        self.limits = kwargs.get("limits")


    
    def alg_stn_bin(self):
        algorithm = cl_alg_stn_bin(
            funtion=self.funtion,
            population=self.population,
            cant_genes=self.num_genes,
            cant_ciclos=self.num_cycles,
            selection_percent=self.selection_percent,
            crossing=self.crossing,
            mutation_percent=self.mutation_percent,
            i_min=self.i_min,
            i_max=self.i_max,
            optimum=self.optimum,
            num_variables=self.num_variables,
            select_mode=self.select_mode
        )
        return algorithm.run()

    def alg_quantum(self):
        algorithm = cl_alg_quantum(
            funtion=self.funtion,
            population=self.population,
            num_qubits=self.num_qubits,
            cant_ciclos=self.num_cycles,
            mutation_percent=self.mutation_percent,
            i_min=self.i_min,
            i_max=self.i_max,
            optimum=self.optimum
        )
        return algorithm.run()
    
    def alg_eda(self):
        algorithm = cl_alg_eda(
            datos = self.data,
            population=self.population,
            num_variables=self.num_variables,
            num_ciclos=self.num_ciclos,
            i_min=self.i_min,
            i_max=self.i_max,
            possibility_selection=self.possibility_selection,
            metric=self.metric,
            model=self.model
        )
        return algorithm.run()
    
    def alg_gp(self): # Genetic programming algorithm
        algorithm = cl_alg_gp(
            data = self.data,
            population = self.population,
            num_ciclos = self.num_ciclos,
            max_depth = self.max_depth
        )
        return algorithm.run()
    

    def alg_de(self): # Differential Evolution algorithm
        algorithm = cl_alg_de(
            function=self.funtion,
            population_size=self.population,
            num_variables=self.num_variables,
            mutation_factor=self.mutation_percent,
            crossover_rate=self.crossing,
            generations=self.num_cycles,
            limits=self.limits,
            optimum=self.optimum    
        )
        return algorithm.run()