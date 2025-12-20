import numpy as np
import random
import operator
import copy
from typing import List, Callable, Tuple
import matplotlib.pyplot as plt

# ==================== NODO DEL ÁRBOL ====================
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []
    
    def evaluate(self, x):
        """Evalúa el árbol de expresión para un valor x dado"""
        if isinstance(self.value, (int, float)):
            return self.value
        elif self.value == 'x':
            return x
        elif self.value in OPERATORS:
            op_func = OPERATORS[self.value]
            child_values = [child.evaluate(x) for child in self.children]
            try:
                return op_func(*child_values)
            except (ZeroDivisionError, OverflowError, ValueError):
                return 1e10  # Penalización por error
        return 0
    
    def copy(self):
        """Crea una copia profunda del árbol"""
        new_children = [child.copy() for child in self.children]
        return Node(self.value, new_children)
    
    def to_string(self):
        """Convierte el árbol a una expresión matemática legible"""
        if isinstance(self.value, (int, float)):
            return str(round(self.value, 2))
        elif self.value == 'x':
            return 'x'
        elif self.value in OPERATORS:
            if len(self.children) == 2:
                left = self.children[0].to_string()
                right = self.children[1].to_string()
                return f"({left} {self.value} {right})"
            elif len(self.children) == 1:
                child = self.children[0].to_string()
                return f"{self.value}({child})"
        return ""
    
    def get_size(self):
        """Retorna el número total de nodos en el árbol"""
        return 1 + sum(child.get_size() for child in self.children)
    
    def get_all_nodes(self):
        """Retorna una lista con todos los nodos del árbol"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

# ==================== OPERADORES Y TERMINALES ====================
def protected_div(a, b):
    """División protegida para evitar división por cero"""
    return a / b if abs(b) > 0.001 else 1

OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': protected_div,
}

TERMINALS = ['x', lambda: random.uniform(-5, 5)]  # Variable x y constantes aleatorias

# ==================== GENERACIÓN DE ÁRBOLES ====================
def create_random_tree(max_depth, method='grow'):
    """Crea un árbol de expresión aleatorio"""
    if max_depth == 0:
        # Crear terminal
        terminal = random.choice(TERMINALS)
        if callable(terminal):
            return Node(terminal())
        return Node(terminal)
    
    if method == 'grow':
        # Método grow: puede elegir terminal o función
        if random.random() < 0.3:
            terminal = random.choice(TERMINALS)
            if callable(terminal):
                return Node(terminal())
            return Node(terminal)
    
    # Crear nodo de función
    op = random.choice(list(OPERATORS.keys()))
    num_children = 2  # Todos nuestros operadores son binarios
    children = [create_random_tree(max_depth - 1, method) for _ in range(num_children)]
    return Node(op, children)

def create_population(pop_size, max_depth):
    """Crea una población inicial usando ramped half-and-half"""
    population = []
    for i in range(pop_size):
        depth = random.randint(2, max_depth)
        method = 'grow' if i % 2 == 0 else 'full'
        population.append(create_random_tree(depth, method))
    return population

# ==================== FUNCIÓN DE FITNESS ====================
def fitness(individual, X, y):
    """Calcula el error cuadrático medio entre predicción y valores reales"""
    predictions = []
    for x_val in X:
        pred = individual.evaluate(x_val)
        if not np.isfinite(pred):
            return 1e10
        predictions.append(pred)
    
    mse = np.mean((np.array(predictions) - y) ** 2)
    # Penalización por complejidad (parsimonia)
    complexity_penalty = individual.get_size() * 0.01
    return mse + complexity_penalty

# ==================== SELECCIÓN ====================
def tournament_selection(population, fitnesses, tournament_size=3):
    """Selección por torneo"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
    return population[winner_idx]

# ==================== OPERADORES GENÉTICOS ====================
def crossover(parent1, parent2):
    """Cruce de subárboles entre dos padres"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Seleccionar nodos aleatorios
    nodes1 = child1.get_all_nodes()
    nodes2 = child2.get_all_nodes()
    
    if len(nodes1) > 1 and len(nodes2) > 1:
        node1 = random.choice(nodes1[1:])  # Evitar raíz
        node2 = random.choice(nodes2[1:])
        
        # Intercambiar subárboles
        node1.value, node2.value = node2.value, node1.value
        node1.children, node2.children = node2.children, node1.children
    
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    """Mutación: reemplaza un subárbol con uno nuevo"""
    if random.random() < mutation_rate:
        nodes = individual.get_all_nodes()
        if len(nodes) > 1:
            node_to_mutate = random.choice(nodes)
            new_subtree = create_random_tree(max_depth=2)
            node_to_mutate.value = new_subtree.value
            node_to_mutate.children = new_subtree.children
    return individual

# ==================== ALGORITMO PRINCIPAL ====================
def genetic_programming(X, y, pop_size=100, generations=50, max_depth=5):
    """Algoritmo de Programación Genética"""
    population = create_population(pop_size, max_depth)
    best_individual = None
    best_fitness = float('inf')
    history = []
    
    for gen in range(generations):
        # Evaluar fitness
        fitnesses = [fitness(ind, X, y) for ind in population]
        
        # Actualizar mejor individuo
        gen_best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()
        
        history.append(best_fitness)
        
        print(f"Generación {gen+1}/{generations} - Mejor fitness: {best_fitness:.6f}")
        
        # Crear nueva población
        new_population = []
        
        # Elitismo: mantener el mejor individuo
        new_population.append(best_individual.copy())
        
        while len(new_population) < pop_size:
            # Selección
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Cruce
            if random.random() < 0.9:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutación
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
    
    return best_individual, history

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    # Generar datos de entrenamiento: f(x) = x^2 + 2x + 1
    np.random.seed(42)
    X_train = np.linspace(-5, 5, 50)
    y_train = X_train**2 + 2*X_train + 1 + np.random.normal(0, 0.5, 50)
    
    print("="*60)
    print("PROGRAMACIÓN GENÉTICA - REGRESIÓN SIMBÓLICA")
    print("="*60)
    print(f"Objetivo: Encontrar f(x) ≈ x² + 2x + 1")
    print(f"Datos de entrenamiento: {len(X_train)} puntos")
    print("="*60)
    
    # Ejecutar algoritmo
    best_solution, history = genetic_programming(
        X_train, y_train, 
        pop_size=200, 
        generations=50, 
        max_depth=5
    )
    
    # Resultados
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    print(f"Mejor expresión encontrada:")
    print(f"f(x) = {best_solution.to_string()}")
    print(f"Fitness (MSE): {fitness(best_solution, X_train, y_train):.6f}")
    print(f"Complejidad (nodos): {best_solution.get_size()}")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Convergencia
    ax1.plot(history, linewidth=2)
    ax1.set_xlabel('Generación', fontsize=12)
    ax1.set_ylabel('Mejor Fitness (MSE)', fontsize=12)
    ax1.set_title('Evolución del Fitness', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Ajuste del modelo
    X_test = np.linspace(-5, 5, 100)
    y_test = X_test**2 + 2*X_test + 1
    y_pred = [best_solution.evaluate(x) for x in X_test]
    
    ax2.scatter(X_train, y_train, alpha=0.6, label='Datos de entrenamiento', s=30)
    ax2.plot(X_test, y_test, 'g--', label='Función real: x² + 2x + 1', linewidth=2)
    ax2.plot(X_test, y_pred, 'r-', label='Función encontrada', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title('Comparación: Real vs Predicción', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("="*60)