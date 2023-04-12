==============================
Lecture 14: Agent-based models
==============================
Note Taker: Sean Donker

Before this class you should:

.. include:: prep14.txt

Before next class you should:

.. include:: prep15.txt

* **March 6th, 11:59pm**, systems thinker signup is `here`_.
* **March 14th, 11:59pm**, group final project signup at `this`_ sheet.
* **March 19th, 11:59pm**, Video presentation due. Instructions at this `link`_.

.. _here: https://docs.google.com/spreadsheets/d/19jsdJxfQh7SAeOOXJClLrQrkQ4vJBvtoaUDEdCI56Dk/edit?usp=sharing
.. _this: https://docs.google.com/spreadsheets/d/1hkBnjJfSh1CprZhGOhgMDLwA51hL7ZB3nnVMk6eWhQA/edit?usp=sharing
.. _link: https://bongolearn.zendesk.com/hc/en-us/articles/360005033094-How-to-Complete-Individual-Project

Agent-based models (ABMs)
-------------------------
Agent-based models are a type of computational model used to study the
behaviour of autonomous agents that have a set of predefined rules.

Like cellular automata (CA), we call them "rule-based". Unlike CAs, ABMs are
considerably more general than CA, meaning they tend to be applicable to a wider
array of systems. Unlike CAs, ABMs don't all have identical rules. It is common
to include randomness in their characteristics.
ABMs are **not** a subclass of CAs.

The Agents
----------
Agents intend to model people or other life (Boids), their traits include:

#. Gathering information from the outside world
#. Using that information to make decisions on how to act
#. Performing some type of action

Agents are usually situated in a space or in a network, meaning they interact
with other agents locally. Agents have imperfect, local information.
This stems from `Bounded_Rationality`_. Bounded Rationality will be visible as
the agent can only 'see' within a small radius of itself.

.. _Bounded_Rationality: https://boycewire.com/bounded-rationality-definition/

Real World Use Example
----------------------
#. Study the effects on price dynamics of low or high-frequency trading

	* Found that high-frequency trading increased market volatility

#. Wildlife ecology and management studies commonly use ABMs to model organisms

	* Used at the University of Guelph for plant pollination

#. Crowd simulation for urban planning. Studying the flow of people

	* Using GPUs to increase the speed and therfore, the model complexity

Dynamic Models of Segregation
-----------------------------
Thomas Schelling introduced a simple model of racial segregation published by
Schelling called Schelling's Model of Segregation. It consists of a 2D grid
where each grid cell represents a house that can be occupied by an agent.
This model has two types of agents that have different colours as their
distinguishing factor. Agents aren't programmed to be racist. Instead, they
prefer to have neighbours of their own kind (colour) in their 8 viewable cells.
If they are happy with their type and the number of neighbours, they will stay.
If they are unhappy, they will move.

Are the Agents Racist?
======================
It is easy to look at the outcome of the simulation and assume the agents are
programmed to be racist. However, most agents would be happy in a mixed
neighbourhood. The **slight preference** towards neighbours of its own kind
creates this sudden shift in results we now expect of complex systems. This
model isn't meant to be used for city planning or anything applicable.
Instead, it is intended to show **why** segregation might show up.

Coding the Model
----------------

Class Initialization
====================
In this implementation, we are inheriting from the class used for CAs to build
the ABM. Again, **ABMs themselves aren't a type of CA!**

Passed init parameters:

* ``n`` is the side length of the square grid world
* ``p`` is the threshold fraction of similar neighbours to upset an agent

.. code-block:: python

  class Schelling(Cell2D):
      def __init__(self, n, p):
          self.p = p # 0 is empty, 1 is red, 2 is blue
          choices = np.array([0, 1, 2], dtype=np.int8)
          probs = [0.1, 0.45, 0.45]
          self.array = np.random.choice(choices, (n, n), p=probs)

``choices`` is an array of different states each cell can be in. ``0`` is empty,
``1`` is red, and ``2`` is blue. ``probs`` is the starting probability of what
type each cell will be initialized too. Meaning roughly 10% of the cells will
be empty. Notice this initialization creates a **random** ``n*n`` world.


Step Function
=============
The step function is how the simulation is updated at each time step (being a
discrete system). For each step, each agent's happiness is computed and saved
in order to update the world in the simulation core.

.. code-block:: python

    options = dict(mode='same', boundary='wrap')
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)

``options`` is a dictionary that is passed to ``correlate2d()``. ``wrap``
ensures the top and bottom, and each side are connected. ``kernel`` defines
what counts as a 'neighbour' to each cell.

.. code-block:: python

    def count_neighbors(self):
        a = self.array
        empty = a==0
        red = a==1
        blue = a==2

        num_red = correlate2d(red, self.kernel, **self.options)
        num_blue = correlate2d(blue, self.kernel, **self.options)
        num_neighbors = num_red + num_blue

        frac_red = num_red / num_neighbors
        frac_blue = num_blue / num_neighbors

        frac_red[num_neighbors == 0] = 0
        frac_blue[num_neighbors == 0] = 0

        frac_same = np.where(red, frac_red, frac_blue)
        frac_same[empty] = np.nan
        return empty, frac_red, frac_blue, frac_same

``empty``, ``red``, and ``blue``, are all boolean arrays of whether each cell in
the array is its corresponding type. Each cell runs ``correlate2d()`` for each
type of neighbour. This is how that agent 'views' the world and makes a
decision. ``np.where(red, frac_red, frac_blue)`` tests the ``red`` boolean and
returns ``frac_red`` if ``True`` and ``frac_blue`` if ``False``. ``frac_same``
will then equal the fraction of like neighbours of that cell. Then used to test
the agent's happiness.

.. code-block:: python

    with np.errstate(invalid='ignore'):
        unhappy = frac_same < self.p
    unhappy_locs = locs_where(unhappy)

The happiness of the agent is then computed to find whether it should move or
not. ``unhappy_locs`` is a set of all of the unhappy agent locations.

Simulation Core
===============
This code loops through all previously determined unhappy agents and moves
them around to open locations.

.. code-block:: python

        num_empty = np.sum(empty)
        for source in unhappy_locs:
            i = np.random.randint(num_empty)
            dest = empty_locs[i]
            # move
            a[dest] = a[source]
            a[source] = 0
            empty_locs[i] = source

``i`` is the index to a random empty cell where an unhappy agent will move.
``dest`` is similar to ``i``, but has the row and column values for the empty
cell. The unhappy agent is moved and then the old occupied cell is set to ``0``
to represent how it is now empty. ``empty_locs[i] = source`` updates the list
of which cells are empty.

For a more quantitative analysis, the ``segregation()`` function computes the
average fraction of similar neighbours (basically how segregated the world is).
This is helpful in plotting charts with varying starting conditions and other
such experiments.

Sugarscape Model
----------------

This ABM was proposed in 1996 to simulate an 'artificial society'. Its main
focus was to support experiments related to **economics** and other social
sciences.

In its simplest form, it is a model of **simple economics**:

* Agents move around on a 2D grid, harvesting and accumulating sugar
* Wealth is represented as sugar. More wealth, more sugar
* Some parts of the grid produce more sugar than others
* Some agents are better at finding sugar than others
* Used to look at **the distribution of wealth**

Sugarscape Landscape
====================
Like the model for segregation, the world is modelled as a 2D grid where agents
can move to different cells. Each cell has a capacity, meaning it can only hold
so much sugar in it. In the *original* configuration, there are two high-sugar
regions with a capacity of 4 followed by outer rings with increasing lower
capacities.

Agent Attributes
================
Each agent has randomly chosen attributes meant to model variations in people:

* **Sugar**: Agents start with a set amount of sugar.
* **Metabolism**: Each agent consumes a set amount of sugar per step.
* **Vision**: How far the agent can detect the amount of sugar in nearby cells.

Agent Movement
==============
Each agent follows a set of rules for moving. It surveys *k* cells in each of
the 4 compass directions. *k* being the range of the agent's vision. It
**chooses** the **unoccupied** cell with the **most sugar**. The agent moves to
the selected cell and **harvests the sugar**, adding the harvest to its
accumulated wealth and leaving the cell empty. It consumes some part of its
wealth, depending on the agent's metabolism. If the resulting total for the
agent's wealth is **negative**, it starves and and is **removed**.

Once all agents have executed the listed steps, the cells **grow back sugar**.
Typically 1 unit, but the capacity is bounded depending on the cell.

The heavy tailed distribution of the wealth seen in this model is common
for most countries in the world.

Coding The Sugarscape Model
---------------------------

In these ABM models with 2d worlds, we tend to go back and forth between pairs
of row column indices and the world grid.

.. code-block:: python

    def make_locs(n, m):
        t = [(i, j) for i in range(n) for j in range(m)]
        return np.array(t)

``make_locs(n, m)`` creates an array with indices from the dimensions ``n``
and ``m`` of the grid.

.. code-block:: python

    def make_visible_locs(vision):
      def make_array(d):
          a = np.array([[-d, 0], [d, 0], [0, -d], [0, d]])
          np.random.shuffle(a)
          return a

      arrays = [make_array(d) for d in range(1, vision+1)]
      return np.vstack(arrays)

``make_visible_locs()`` returns the indices of the cells that are visible given
the ``vision`` attribute of the agent. Position is relative here, you don't need
to specify the location.

.. code-block:: python

  def distances_from(n, i, j):
      X, Y = np.indices((n, n))
      return np.hypot(X-i, Y-j)

``distances_from()`` will return the relative distances to different cells from
the specified location. ``n`` is the size of the array, (``n*n``). ``i`` and
``j`` are the coordinates for where you want the distances measured from.


.. code-block:: python

    bins = [3, 2, 1, 0]
    np.digitize(dist, bins)

This helper code digitizes the distances made by ``distances_from()`` but in
the opposite direction. The farther distances will be assigned lower values and
closer distances have higher values. Makes the distances discrete.

Sugarscape Notes
================
The Sugarscape class is, again, inheriting from the Cell2D class, but it is used
for the world or the environment instead. Worlds tend to have a set
**carrying capacity** that depends on the world parameters. This value is the
number of agents the world can support at equilibrium.

Agents are programmed to die after a given amount of steps even if they have
enough sugar to consume. This is needed in order to plot the distribution of
wealth over the lifetime of the agents. We can then see the expected
long-tailed distribution CDF (cumulative distribution function) plots. This
basically means most agents die with a small amount of wealth, but some die with
a lot.

Emergent Behavior
=================
If you start all the agents in a certain corner, it is possible to see them
**propagate** over across the world on a **diagonal**. This is important
because the individual agents **can't** actually move **diagonally** themselves!
This is called a **collective behaviour**.

Conclusions
-----------
**Emergent properties** is a characteristic of a complex system that results
from the interactions of its components, not their properties.
Some examples of emergent properties we have seen so far include:

#. **Segregation** in Schelling's model.

	* It is not caused by programming the agents to be racist.

#. **Distribution of Wealth** in the Sugarscape model.

	* A weaker example but wasn't modelled to have the resulting wealth distribution

#. **Diagonal Wave Behaviour** in the Sugarscape model.

	* The agents were clearly not programmed to move diagonally on their own.

**Emergent properties** are a characteristic of complex systems! Large language
models (like chatGPT), which are very popular currently, have been seen doing
things that, originally, were not designed into the system!
