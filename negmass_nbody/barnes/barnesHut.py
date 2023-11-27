# Barnes hut module
# Gerrit Fritz

import numpy as np
import math


class Node():
    """Stores data for Octree nodes."""

    def __init__(self, middle, dimension):
        """Method that sets up a node.

        Method that sets up and declares variables for an octree node.

        Args:
            middle: Position of center of the position.
            dimension: Length of sides of a node.
        """
        self.particle   = None
        self.middle     = middle
        self.dimension  = dimension
        self.mass       = None
        self.corners    = None
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
        """Method that creates a subnode.

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
        x, y, z = 1, 1, 1
        if point[0] > self.middle[0]:  # right
            x = 0
        if point[1] > self.middle[1]:  # front
            y = 0
        if point[2] > self.middle[2]:  # top
            z = 0
        return [x, y, z]

    def get_corners(self):
        """Method that gets corners of a node.

        Method that get corners of a node for visualization. Iterates through
        the top and bottom for front and back for right and left.
        """
        if self.corners is None:
            self.corners = []
            for x in [1, -1]:          # right or left
                for y in [1, -1]:      # front or back
                    for z in [1, -1]:  # top or bottom
                        pos = [self.middle[0] + ((self.dimension / 2) * x),
                               self.middle[1] + ((self.dimension / 2) * y),
                               self.middle[2] + ((self.dimension / 2) * z)]
                        self.corners.append(pos)
        return self.corners

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

    def __init__(self, particles, root_node, theta, node_type):
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
        self.node_type = node_type

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

    def update_forces_collisions(self):

        self.collision_dic = {}
        for particle in self.particles:
            self.collision_dic[particle] = []
            particle.force = np.array([0., 0., 0.])
            particle.e_pot = 0
            self.calc_forces(self.root_node, particle)
            particle.e_pot /= 1  # 2

            if len(self.collision_dic[particle]) == 0:
                #print('Debe ser 0: ',len(self.collision_dic[particle]))
                del self.collision_dic[particle]

    def calc_forces(self, node, particle):
        """Method that calculates the force on an octree particle.

        Method that calculates the force on an octree particle by iterating
        through the octree.
        If the node is a point node that doesnt hold the body itelf, directly
        calculate the forces.
        If its a regional node and the dimension/distance ratio is smaller
        than theta and the center of mass position is not the same as the
        particle position, calculate the force between the node and
        the particle.

        Args:
            node: Quadrant of node.
            particle: Simulation body.
        """
        if node.particle is not None and node.particle != particle:
            force, e_pot, distance = self.gravitational_force(particle, node, np.array([]), np.array([]))
            particle.force -= force
            particle.e_pot -= e_pot
            # if distance < particle.radius + node.particle.radius and particle.mass > node.particle.mass:
            #     self.collision_dic[particle].append(node.particle)

        elif node.particle is None and not np.array_equal(particle.position, node.center_of_mass):
            distance = np.array([*particle.position]) - np.array([*node.center_of_mass])
            r = math.sqrt(np.dot(distance, distance))
            d = node.dimension
            if d / r < self.theta:
                force, e_pot, distance = self.gravitational_force(particle, node, distance, r)
                particle.force -= force
                particle.e_pot -= e_pot
            else:
                for subnode in node.nodes:
                    self.calc_forces(subnode, particle)

    def gravitational_force(self, particle, node, distance_vec, distance_val):  # can be ragional or point node
        """Method that calculates the force between two particles.
    
        Method that calculates the force acted on the particle by
        another particle or a node. Only calculates the distance and vector
        did not have to be calculated for theta.
    
        Args:
            particle: Simulation body.
            node: Node of the octree.
            distance_vec: Vector of distance between the bodies.
            distance_val: Magnitude of distance betweent the bodies
        
        Returns:
            The force, potential energy and distance between two bodies or 
            the body and the node.
        """
        force = np.array([0., 0., 0.])
        if len(distance_vec) == 0 and len(distance_val) == 0:
            distance = np.array([*particle.position]) - np.array([*node.center_of_mass])
            distance_mag = math.sqrt(np.dot(distance, distance))
        else:
            distance = distance_vec
            distance_mag = distance_val

        G = 1 #G igual a 1
        e_pot = G * particle.mass * node.mass / distance_mag
        force_mag = G * particle.mass * node.mass / np.dot(distance, distance)
        force = (distance / distance_mag) * force_mag
        return force, e_pot, distance_mag

    def get_all_nodes(self, node, lst):

        if node.particle is None and len(node.nodes) >= 1 or node.particle is not None:
            if len(node.nodes) >= 1:
                if self.node_type == "regional" or self.node_type == "both":
                    lst.append(node.get_corners()) 
                for subnode in node.nodes:
                    self.get_all_nodes(subnode, lst)
            if node.particle is not None and (self.node_type == "point" or self.node_type == "both"):
                lst.append(node.get_corners())


# if __name__ == "__main__":
#     from Simulation import Planet
#     planet1 = Planet
#     planet1.position = (10, 20, 30)
#     planet1.mass = 200

#     planet2 = Planet
#     planet2.position = (-10, -20, -30)
#     planet2.mass = 20

#     data = [planet1, planet2]  # planet2]
#     root = Node([0, 0, 0], 100)
#     theta = 1

#     print(root.in_bounds(planet2.position))
#     quad = root.get_quad(planet2.position)
#     print(quad)
#     root.create_subnode(quad)
#     subnode = root.get_subnode(quad)
#     print(subnode.middle)
#     print(subnode.get_corners())


class Simulation:
    """Handles data and connection between simulation bodies."""

    def __init__(self, theta=1, rc=0, absolute_pos=True, focus_index=0):
        """Setup of the simulation.

        Method that sets up the simulation with parameters and type of
        focus.

        Args:
            theta: Theta value for the Barnes Hut simulation.
            rc: Restitution coefficient for collisions.
            absolute_pos: Bool value to determine type of movement.
            focus_index: Index of the list focus_options form 0 to 2.
            node_type: String that determines what nodes are displayed.
        """
        self.restitution_coefficient = rc
        self.focus_options  = ["none", "body", "cm"]
        self.absolute_pos   = absolute_pos
        self.theta          = theta
        self.iteration      = 0
        self.bodies         = []
        self.focused_body   = None
        self.first          = True
        self.time           = 0
        self.total_ekin     = 0
        self.total_epot     = 0
        self.total_e        = 0
        self.cm_pos         = np.array([0, 0, 0])
        self.cm_velo        = None
            
        if focus_index >= 0 and focus_index < len(self.focus_options):
            self.focus_index = focus_index
        else:
            self.focus_index = 0
            print(f"focus index {focus_index} not in focus options {self.focus_options}, swithing to default {self.focus_options[0]}")
        self.focus_type = self.focus_options[focus_index]

    def add_body(self, body):
        self.bodies.append(body)

    def get_bodies(self):
        return self.bodies

    def remove_body(self, body):
        self.bodies.remove(body)
        if body == self.focused_body and len(self.bodies) > 0:
            self.focused_body = random.choice(self.bodies)
        elif body == self.focused_body:
            print("error: no bodies left")

    def set_focus(self, body):
        if body in self.bodies:
            self.focused_body = body
        elif self.focus_type == "body":
            self.focused_body = random.choice(self.bodies)
        else:
            self.focused_body = None

    def update_center_of_mass(self, timestep):
        new_pos = np.array(self.tree.root_node.center_of_mass)
        old_pos = self.cm_pos
        self.cm_velo = (new_pos - old_pos) / timestep
        self.cm_pos = new_pos

    def get_focus_pos(self):
        if self.focus_type == "body":
            pos = self.focused_body.position
        elif self.focus_type == "cm":
            pos = self.cm_pos
        else:
            pos = np.array([0, 0, 0])
        return pos

    def switch_focus(self, direction):
        if self.focus_type == "body":
            focused_index = self.bodies.index(self.focused_body)

            if direction == "previous":
                focused_index += 1
                if focused_index > len(self.bodies) - 1:
                    focused_index = 0

            elif direction == "next":
                focused_index -= 1
                if focused_index < 0:
                    focused_index = len(self.bodies) - 1

            self.set_focus(self.bodies[focused_index])

    def clear_trail(self):
        for body in self.bodies:
            body.trail = []

    def calculate(self, timestep, draw_box, node_type):
        """Method that calculates a simulation physics step.

        Method that calls functions for physics calculations. 
        Also includes caclulations for the total energy. If draw_box is
        true the boxes of the octree are also extracted.

        Args:
            timestep: Amount of seconds that are counted with in the
                physics calculations.
            draw_box: Boolean value that determines if cube data should be 
                extracted. Implemented to run faster.
        """
        if self.first:
            self.first = False
            self.update_interactions(node_type)
            for body in self.bodies:
                body.acceleration = body.force / body.mass
        

        self.update_center_of_mass(timestep)
        for body in self.bodies:
            body.update_velocity(timestep)
            body.update_position(timestep)

        return
        self.update_interactions(node_type)

        self.tree_nodes = []
        if draw_box:
            self.tree.get_all_nodes(self.tree.root_node, self.tree_nodes)

        for body in self.destroyed:
            self.remove_body(body)

        self.total_ekin = 0
        self.total_epot = 0
        self.total_e = 0
        for body in self.bodies:
            self.total_ekin += body.e_kin
            self.total_epot += (body.e_pot / 2)  # no clue if this is right
        self.total_ekin /= len(self.bodies)
        self.total_epot /= len(self.bodies)

        self.total_e = self.total_ekin + self.total_epot  # epot is negative, higher energy the further you are

        self.time += timestep
        self.iteration += 1

    def get_data(self):
        """Method that gets simulation data.

        Method that exports simulation data in the form of a list.
        The list includes almost all values of the simulation excluding the
        bodies themselves.

        Returns:
            Body data, Octree data and system data for further usage.
        """
        default_pos = self.get_focus_pos()

        body_data = []
        for body in self.bodies:
            body_type = type(body)
            name    = body.name
            pos     = body.position - default_pos
            radius  = body.radius
            mass    = body.mass
            color   = body.color
            trail = [position - default_pos for position in body.trail]
            body_data.append((name, pos, radius, mass, color, trail, body.velocity,
                              body.acceleration, body.density, body.force,
                              body.e_kin, body.e_pot, body_type))

        for cube in self.tree_nodes:
            for point in cube:
                point[0] -= default_pos[0]
                point[1] -= default_pos[1]
                point[2] -= default_pos[2]

        system_data = [self.focus_type, self.focused_body, self.absolute_pos,
                       self.theta, self.restitution_coefficient, self.total_ekin,
                       self.total_epot, self.total_e, self.cm_pos, self.iteration,
                       len(self.bodies), self.tree.root_node.middle, self.tree.root_node.dimension]

        return body_data, self.tree_nodes, system_data, self.cm_pos - default_pos

    def update_interactions(self, node_type):
        center = self.get_focus_pos()

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
        self.tree = Octree(self.bodies, root, self.theta, node_type)
        root.compute_mass_distribution()
        self.tree.update_forces_collisions()
        self.compute_collisions(self.tree.collision_dic)

    def compute_collisions(self, collisions):#Ojito.
        """Method that computes body collisions.

        Method that computes body collisions based on the resitution
        coefficient. Sort collisions from lowest to highest mass. So all
        collisions can be determined. Calls body to body inelastic collision.

        Args:
            collisions: All collisons as extractes from the Barnes-Hut program.
        """
        self.destroyed = []
        bodies = list(collisions.keys())
        bodies.sort(key=lambda element: element.mass)
        for body in bodies:
            other_bodies = collisions[body]
            for other in other_bodies:
                if other not in self.destroyed:
                    body.inelastic_collision(other, self.restitution_coefficient)
    


class SimulationBody:
    """Data storage and caclulation of a simulation body"""
    #def __init__(self, simulation, name, mass, density, position, velocity, color, nr_pos, point_dist):
    
    def __init__(self, simulation, mass, position, velocity, nr_pos=50, point_dist=10):
        """Method that sets up a simulation body.

        Method sets up and declares variables for a simulation body.

        Args:
            simulation: Simulation that stores all other bodies.
            name: Name/Unique identifier of body.
            mass: Mass of the body in kg.
            density: Density of the body in g/cm3.
            position: Position vector in meters.
            velocity: Velocity vector in m/s.
            color: Color of the body.
            nr_pos: Trail node number of the body.
            point_dist: Distance between nodes in the trail.
        """
        self.simulation     = simulation
        #self.name           = name
        self.mass           = mass  
        # self.density        = density  # density in g/cm^3
        # self.radius         = self.calculate_radius()
        self.position       = np.array([*position])
        self.velocity       = np.array([*velocity])
        # self.color          = color
        self.nr_pos         = nr_pos
        self.counter        = 0
        self.point_dist     = point_dist
        self.trail          = []
        self.acceleration   = np.array([0, 0, 0])
        self.force          = np.array([0, 0, 0])
        self.e_pot          = 0
        self.e_kin          = 0
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

        if not self.simulation.absolute_pos:  # relative movement
            if self.simulation.focus_type == "body":
                self.position -= (timestep * (self.simulation.focused_body.velocity + timestep * self.simulation.focused_body.acceleration / 2))
            elif self.simulation.focus_type == "cm":
                self.position -= self.simulation.cm_velo

        self.counter += 1
        if self.counter == self.point_dist:
            self.trail.append(self.position.copy())
            self.counter = 0

        if len(self.trail) > self.nr_pos:
            del self.trail[0]

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
        self.e_kin = .5 * self.mass * np.dot(self.velocity, self.velocity)

    # def calculate_radius(self):
    #     density_kg = self.density * 1000
    #     return ((3 * self.mass) / (4 * math.pi * density_kg))**(1 / 3)

    def inelastic_collision(self, other, restitution_coefficient):
        """Method that calculates an inelastic collision.

        Method that calculates the collision between two objects based
        on the restitution coefficient. If that coefficient if 0 the planets
        merge. If the coefficient is higher than 0 to 1 the collision is more
        elastic. Destroys body that is smaller than the other.

        Args:
            other: Other body that takes part in the collision.
            restitution_coefficient: Coefficient that determines type of
                collision.
        """
        # only if restitution_coefficient is 0 bodies merge
        velo_u = ((self.mass * self.velocity) + (other.mass * other.velocity)) / (self.mass + other.mass)

        if restitution_coefficient == 0:
            # merge of planets (other is destroyed)
            self.velocity = velo_u
            self.mass += other.mass
            self.radius = self.calculate_radius()
            self.simulation.destroyed.append(other)
        else:
            # somewhat elastic collision
            r = restitution_coefficient
            self.velocity = ((self.velocity - velo_u) * r) + velo_u
            other.velocity = ((other.velocity - velo_u) * r) + velo_u

