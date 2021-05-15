## Simulation of a Chemical Reactor
The project mainly focused on identifying a non-linear system of neutralisation reactor and designing an appropriate control strategy.
The repository includes ready-made simulations of a chemical reactor with standard identification and deep learning models in both open-loop and closed-loop form. 

### Prerequisites
It is necessary to install python 3.7.6+ with the required packages listed in *requirement.txt* file.
Also, it is recommended to create a separate environment for the project to avoid unexpected collisions.
In order to install the required packages, just run:

	    python -m pip install -r requirements.txt
   
### Ready-Made Scripts:

 
* Open-loop simulation with:
    *  Numerical method
        * num_open_loop.py
    * Neural network (FFNN, RNN, LSTM)
        * nn_open_loop.py
    * ARX model
        * ARX.py
    * Strejc method
        * Strejc.py
    * Analytic solution
        * analytic_solution.py
    

* Closed-loop simulation with:   
    *  Numerical method
        * num_closed_loop.py 
    * Neural network (FFNN, RNN, LSTM)
        * nn_closed_loop.py
    * Verification of neural network with numerical method
        * num_nn_closed_loop.py


### Source packages:

* simulation
    - Contains libraries for the model, open-loop and closed-loop simulation with numerical 
        methods and deep learning models too.
* identification
    - Contains libraries with standard identification models, including ARX, Strejc method.



