# Hotelling Spatial Model – Theoretical Background

## 1. The Classic 1-D Hotelling Model (1929)

Harold Hotelling (1929) introduced a seminal model of spatial competition.
Two firms choose locations on a unit-length "Main Street" and then set prices.
Consumers are uniformly distributed along the street and purchase from the firm
that minimises their *full price* (product price + transport cost).

### Key results

| Result | Description |
|--------|-------------|
| **Principle of Minimum Differentiation** | Under linear transport costs and simultaneous location choice, both firms converge to the centre of the market. |
| **Price-location game** | When location *and* price are chosen simultaneously the pure-strategy Nash equilibrium may not exist under linear transport costs (d'Aspremont et al., 1979). |
| **Quadratic transport costs** | With quadratic transport costs, a stable equilibrium exists at maximum differentiation (firms locate at the two endpoints). |

## 2. Extension to 2 Dimensions

The 2-D generalisation replaces the unit interval with a compact market region
(e.g. a unit square or a real geographic area).  Consumers are distributed
according to a density function *f(x, y)* which may be uniform (synthetic
city) or derived from real population data.

The delivered price for consumer at *(x_c, y_c)* buying from firm *i* at
*(x_i, y_i)* with price *p_i* is:

$$C_i(x_c, y_c) = p_i + t \cdot d\bigl((x_c, y_c),\,(x_i, y_i)\bigr)$$

where *t* is the transport cost parameter and *d(·,·)* is typically the
Euclidean distance.

## 3. LLM and Reinforcement-Learning Agents

This toolkit replaces the classical "rational agent" assumption with two
alternative decision-making paradigms:

### 3a. Large Language Models (LLM agents)
The firm's pricing and location strategy is delegated to an LLM.  At each
period the model receives a natural-language description of the market state
(competitor positions, prices, its own market share) and returns a JSON action
`{"location": [x, y], "price": p}`.

### 3b. Q-Learning Agents
The firm learns a tabular Q-function over discretised (location, price) states
using standard ε-greedy Q-learning.  The reward signal is the firm's profit at
each period.

## 4. References

* Hotelling, H. (1929). *Stability in Competition*. The Economic Journal, 39(153), 41–57.
* d'Aspremont, C., Gabszewicz, J. J., & Thisse, J.-F. (1979). *On Hotelling's 'Stability in Competition'*. Econometrica, 47(5), 1145–1150.
* Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
