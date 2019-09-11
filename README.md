**MONOD Parameter Estimator**
-

Simple Tool for Parameter Estimation according based on the Monod Growth Kinetics including an estimation of Product Formation according to the Luedeking-Piret equation:

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

The data file can be specified in *config.py*. Please provide data in the form of a *.csv*-file containing the following headers:

* t = time
* X = biomass
* S = substrate
* P = product

Literature:
* Jacques Monod. The Growth of Bacterial Cultures. Ann Rev Microbiol., 1949; 3:371.
* Luedeking R, Piret EL. A kinetic study of the lactic acid fermentation. Batch process at controlled pH. J Biochem Microbiol. 1959; 1:393–412.