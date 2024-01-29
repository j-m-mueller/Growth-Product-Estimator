**MONOD Parameter Estimator**
-

Repository for Parameter Estimation according based on the Monod Growth Kinetics including an estimation of Product Formation according to the Luedeking-Piret equation:

* dX/dt = µ * X * S/(KS+S) 
* dS/dt = dX/dt * -1/Y(X/S)
* dP/dt = pX * X + pµ * dX/dt

Parameters:
* µ = maximal specific growth rate
* X = biomass concentration
* S = substrate concentration
* KS = Monod constant
* Y(X/S) = yield coefficient for biomass per substrate
* pX = term for biomass-related product formation
* pµ = term for growth-related product formation

Reads data from an input .csv file and estimates parameters based on provided initial estimates and bounds.

Adjustments to paths and parameters can be made via *config.yml*.

The input *.csv* file is expected to have the columns:

* t = time
* X = biomass
* S = substrate
* P = product

Please run the *demonstrator-notebook.ipynb* for a demonstration.

Literature:
* Jacques Monod. The Growth of Bacterial Cultures. Ann Rev Microbiol., 1949; 3:371.
* Luedeking R, Piret EL. A kinetic study of the lactic acid fermentation. Batch process at controlled pH. J Biochem Microbiol. 1959; 1:393–412.


Update Log:

- initial commit: 09/2019
- major refactor, integration of config validation: 01/2024
