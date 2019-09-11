**MONOD Parameter Estimator**
-

Simple Tool for Parameter Estimation according to a simple Monod Growth Kinetics including estimation of Product Formation based on a growth-associated term (*pmu*) and a purely biomass-based term (*pX*) according to the Luedeking-Piret equation. 

The data file can be specified in *config.py*. Please provide data in the form of a *.csv*-file containing the following headers:

* t = time
* X = biomass
* S = substrate
* P = product

Literature:
* Jacques Monod. The Growth of Bacterial Cultures. Ann Rev Microbiol., 1949; 3:371.
* Luedeking R, Piret EL. A kinetic study of the lactic acid fermentation. Batch process at controlled pH. J Biochem Microbiol. 1959; 1:393–412.