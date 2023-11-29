# Barnes hut module
# Gerrit Fritz

import numpy as np
import math


class Node():
    """Stores data for Octree nodes."""

    def __init__(self, middle, dimension):
        """
        Método que crea un nodo.

        Args:
            middle: Position of center of the position.
            dimension: Length of sides of a node.
        """
        self.particle   = None
        self.middle     = middle
        self.dimension  = dimension
        self.mass       = None
        self.center_of_mass = None
        self.nodes = []
        self.subnodes = [[[None, None],      # right front (top, bottom)
                          [None, None]],     # right back (top, bottom)
                         [[None, None],      # left front (top, bottom)
                          [None, None]]]     # left back (top, bottom)

    def insert_particle(self, particle):
        self.particle = particle

    def get_subnode(self, quad):
        return (self.subnodes[quad[0]][quad[1]][quad[2]])

    def create_subnode(self, quad):
        """
        Method that creates a subnode.

        Method that determines the middle and dimension of the subnode
        of a specific quadrant of the node. Initializes a subnode and adds
        that subnode to the nodes.

        Args:
            quad: Quadrant of node.
        """
        dimension  = self.dimension / 2

        x, y, z = 1, 1, 1
        if quad[0] == 1:
            x = -1
        if quad[1] == 1:
            y = -1
        if quad[2] == 1:
            z = -1

        middle = [self.middle[0] + ((dimension / 2) * x),  # value  1, right
                  self.middle[1] + ((dimension / 2) * y),  # value  1, front
                  self.middle[2] + ((dimension / 2) * z)]  # value  1, top
        node       = Node(middle, dimension)
        self.subnodes[quad[0]][quad[1]][quad[2]] = node
        self.nodes.append(node)

    def get_quad(self, point):
        #Solo con 3 valores se puede determinar en cuál de los 4 cuadrantes está el nodo.
        x, y, z = 1, 1, 1
        if point[0] > self.middle[0]:  # Si x= 0, es el nodo que está a la derecha. Si x=1, es el nodo que está a la izquierda.
            x = 0
        if point[1] > self.middle[1]:  # Si x= 0, es el nodo que está en frente. Si x=1, es el nodo que está en el fondo.
            y = 0
        if point[2] > self.middle[2]:  # Si x= 0, es el nodo que está arriba. Si x=1, es el nodo que está abajo.
            z = 0
        return [x, y, z]

    def in_bounds(self, point):
        val = False
        if point[0] <= self.middle[0] + (self.dimension / 2) and\
           point[0] >= self.middle[0] - (self.dimension / 2) and\
           point[1] <= self.middle[1] + (self.dimension / 2) and\
           point[1] >= self.middle[1] - (self.dimension / 2) and\
           point[2] <= self.middle[2] + (self.dimension / 2) and\
           point[2] >= self.middle[2] - (self.dimension / 2):
            val = True
        return val

    def compute_mass_distribution(self):
        """Method that calculates the mass distribution.

        Method that calculates the mass distribution of the node based on
        the mass posistions of the subnode weighted by weights of 
        the subnodes.
        """

        if self.particle is not None:
            self.center_of_mass = np.array([*self.particle.position])
            self.mass = self.particle.mass
        else:
            # Compute the center of mass based on the masses of all child quadrants
            # position based on child quadrants weights with their mass
            self.mass = 0
            self.center_of_mass = np.array([0., 0., 0.])
            for node in self.nodes:
                if node is not None:
                    node.compute_mass_distribution()
                    self.mass += node.mass
                    self.center_of_mass += node.mass * node.center_of_mass
            self.center_of_mass /= self.mass



class Octree():
    """Handles setup and calculations of the Barnes-Hut octree."""

    def __init__(self, particles, root_node, theta):
        """Method that sets up an octree.

        Method that sets up the variables for the octree. Calls functions
        for creation of the octree.

        Args:
            particles: List of particles that are inserted.
            root_node: Root node of the octree.
            theta: Theta that determines the accuracy of the simulations.
        """
        self.theta = theta
        self.root_node = root_node
        self.particles = particles
        

        for particle in self.particles:
            self.insert_to_node(self.root_node, particle)

    def insert_to_node(self, node, particle):
        """Recursive method that inserts particles into the octree.

        Recursive method that inserts bodies into the octree.
        Checks if particle is in the current node to prevent bounds issues.
        Determines the appropriate child node and gets that subnode.
        If that subnode is empty insert the particle and stop.
        If the child node point is a point node (one particle) turn it into 
        a regional node by inserting both particles into it.
        If the child node is a regional node insert the particle.

        Args:
            node: Quadrant of node.
            particle: Simulation body.
        """
        # check if point is in cuboid of present node
        if not node.in_bounds(particle.position) and not np.array_equal(particle.position, self.root_node.middle):
            print("error particle not in bounds")
            print(f"middle: {node.middle}, dimension: {node.dimension}, particle position: {particle.position}, type: {type(particle)}")
            return

        quad = node.get_quad(particle.position)
        if node.get_subnode(quad) is None:
            node.create_subnode(quad)
        subnode = node.get_subnode(quad)

        if subnode.particle is None and len(subnode.nodes) == 0:  # case empty node
            subnode.insert_particle(particle)

        elif subnode.particle is not None:  # case point node
            old_particle = subnode.particle
            subnode.insert_particle(None)
            self.insert_to_node(subnode, old_particle)
            self.insert_to_node(subnode, particle)

        elif subnode.particle is None and len(subnode.nodes) >= 1:  # case region node
            self.insert_to_node(subnode, particle)

    def update_forces(self,G,epsilon):
        for particle in self.particles:
            particle.force = np.array([0., 0., 0.])
            self.calc_forces(self.root_node, particle,G,epsilon)
            

            
    def calc_forces(self, node, particle,G,epsilon):
        """Método que calcula la fuerza sobre una partícula de un octree.

        Método que calcula la fuerza sobre una partícula de un octree al iterar a través del octree. 
        Si el nodo es un nodo puntual que no contiene el cuerpo en sí mismo, calcula directamente las fuerzas.

        Si es un nodo regional y la relación entre la dimensión/distancia es menor que theta 
        y la posición del centro de masa no es la misma que la posición de la partícula, 
        calcula la fuerza entre el nodo y la partícula.

        Args:
            node: Quadrant of node.
            particle: Simulation body.
        """
        
        if node.particle is not None and node.particle != particle:
            force, distance = self.gravitational_force(particle, node, np.array([]), np.array([]),G,epsilon)
            particle.force -= force
            
            
            
        elif node.particle is None and not np.array_equal(particle.position, node.center_of_mass):
            distance = np.array([*particle.position]) - np.array([*node.center_of_mass])
            r = math.sqrt(np.dot(distance, distance))
            d = node.dimension
            if d / r < self.theta:
                force, distance = self.gravitational_force(particle, node, distance, r,G,epsilon)
                particle.force -= force
            else:
                for subnode in node.nodes:
                    self.calc_forces(subnode, particle,G,epsilon)

    def gravitational_force(self, particle, node, distance_vec, distance_val,G,epsilon):  # can be ragional or point node
        """
        Method that calculates the force between two particles.
    
        Method that calculates the force acted on the particle by
        another particle or a node. Only calculates the distance and vector
        did not have to be calculated for theta.
    
        Args:
            particle: Simulation body.
            node: Node of the octree.
            distance_vec: Vector of distance between the bodies.
            distance_val: Magnitude of distance betweent the bodies
        
        Returns:
            The force and distance between two bodies or 
            the body and the node.
        """
        force = np.array([0., 0., 0.])
        if len(distance_vec) == 0 and len(distance_val) == 0:
            distance = np.array([*particle.position]) - np.array([*node.center_of_mass])
            distance_mag = math.sqrt(np.dot(distance, distance))
        else:
            distance = distance_vec
            distance_mag = distance_val
        distance_mag = np.sqrt(np.dot(distance, distance)+np.square(epsilon))
            
        force_mag = G * particle.mass * node.mass / np.dot(distance, distance)
        force = (distance / distance_mag) * force_mag
        return force, distance_mag



class Simulation:
    """Handles data and connection between simulation bodies."""

    def __init__(self,G,epsilon,theta=1/2):
        """Setup of the simulation.

        Method that sets up the simulation with parameters and type of
        focus.

        Args:
            theta: Theta value for the Barnes Hut simulation.
            G: Gravitational constant.
            epsilon: Softening factor.
        """
        
        
        self.theta          = theta
        self.bodies         = []
        self.G=G
        self.epsilon=epsilon
        
            
       

    def add_body(self, body):
        self.bodies.append(body)

    def calculate(self, timestep):
        """
        Método que calcula un paso de simulación.

        Args:
            timestep: Amount of seconds that are counted with in the physics calculations.
        """
        
        self.update_interactions()
        for body in self.bodies:
            body.acceleration = body.force / body.mass
    
        for body in self.bodies:
            body.update_position(timestep)
            body.update_velocity(timestep)
            

        return

        self.update_interactions(node_type)
        
       

    

    def update_interactions(self):
        ##Se establece el centro del nodo raíz. 
        center = np.array([0, 0, 0])

        ##Se calcula cuál es el cuerpo que está más lejos para conocer cuál debe ser la dimensión del Octree.

        largest_val = 0
        furthest_body = None
        for body in self.bodies:
            for i, val in np.ndenumerate(body.position):
                dist_sqr = (val - center[i])**2
                if dist_sqr > largest_val:
                    largest_val = dist_sqr
                    largest_index = i
                    furthest_body = body

        dimension = math.sqrt(((furthest_body.position[largest_index] - center[largest_index]) * 2.5)**2)
        root = Node(center, dimension)
        ##Se crea el Octree:
        self.tree = Octree(self.bodies, root, self.theta)
        ##Se calcula la distribución de masa del Octree.
        root.compute_mass_distribution()
        ##Se actualizan las fuerzas:
        self.tree.update_forces(self.G,self.epsilon)
        

    
    


class SimulationBody:
    """Data storage and caclulation of a simulation body"""
    
    def __init__(self, simulation, mass, position, velocity):
        """Method that sets up a simulation body.

        Method sets up and declares variables for a simulation body.

        Args:
            simulation: Simulation that stores all other bodies.
            mass: Mass of the body in kg.
            position: Position vector in meters.
            velocity: Velocity vector in m/s.
           
           
        """
        self.simulation     = simulation
        self.mass           = mass  
        self.position       = np.array([*position])
        self.velocity       = np.array([*velocity])
        self.acceleration   = np.array([0, 0, 0])
        self.force          = np.array([0, 0, 0])
        self.simulation.add_body(self)

    def update_position(self, timestep):
        """Method that calculates the body position.

        Method that calculates the body position follwing verlet velocity
        integration. Subtracts movement of focus_body if the view is movement
        is relative.

        Args:
            timestep: Amount of seconds for the calculations.
        """
        self.position += (timestep * (self.velocity + timestep * self.acceleration / 2))

    def update_velocity(self, timestep):
        """Method that calculates body velocity.

        Method that calculates body velocity following velocity
        verlet ingtegration.

        Args:
            timestep: Amount of seconds for the calculations.
        """
        newacc = self.force / self.mass
        self.velocity += (self.acceleration + newacc) * timestep / 2
        self.acceleration = newacc
        

   
