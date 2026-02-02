import numpy as np
import copy

class EvolutionManager:
    """
    Handles the genetic algorithm logic for a population of 2 cars.
    Selection: Keep the best performer.
    Mutation: Tweak weights of the offspring.
    """
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate
        self.generation = 1

    def evolve(self, cars):
        """
        Takes a list of cars, finds the best, and returns new brains for the next gen.
        """
        self.generation += 1
        
        # Sort cars by fitness (distance)
        cars.sort(key=lambda x: x.fitness, reverse=True)
        
        # Log best fitness
        print(f"Gen {self.generation-1} Best Fitness: {cars[0].fitness:.1f}")
        
        new_brains = []
        
        # ELITISM: Keep the top 2 best cars exactly as they are
        # This ensures we never lose our best performing logic
        num_elites = 2
        for i in range(num_elites):
            if i < len(cars):
                new_brains.append(copy.deepcopy(cars[i].brain))
                
        # OFFSPRING: Fill the rest of the population with mutated versions of the elites
        # We alternate mutating the best and the second best
        remaining_slots = len(cars) - len(new_brains)
        
        for i in range(remaining_slots):
            parent = cars[i % num_elites] # Alternate parents
            child_brain = copy.deepcopy(parent.brain)
            self.mutate(child_brain)
            new_brains.append(child_brain)
        
        return new_brains

    def mutate(self, brain):
        """Randomly tweaks the weights and biases of a brain."""
        weights = brain.get_weights()
        
        for key in weights:
            # Add small random noise to some weights
            mask = np.random.random(weights[key].shape) < self.mutation_rate
            noise = np.random.uniform(-0.5, 0.5, weights[key].shape)
            weights[key][mask] += noise[mask]
            
        brain.set_weights(weights)
