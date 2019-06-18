[//]: # (Image References)

[image1]: https://github.com/bonniesjli/DQN_SR/blob/master/asset/sr.gif "Trained Agent"

# DQN-SR

This is an implementation of paper "Count Based Exploration with Successor Representation" by Machado et al. 

![Trained Agent][image1]

### TO DOs
- [ ] Hyperparameter tuning
- [ ] Feature vector normalization
- [ ] Add curves and model weights

### Instructions
* Environment<br>
I have provided pre-built environment for Windows and Linuz users.<br>
In `main.py` : `line 29` set the env according to the system you are using.<br>

* Run `python -m main.py` to train the agent 

### Files
* `main.py` - load the environment, explore the environment, train the agent or run the trained agent
* `agent.py` contains the agent class 
* `model.py` contains the neural network models the agents employ. 
* `checkpoint.pth` contains trained model weights <br>
* `REPORT.md` contains description of implementation, results, and ideas for future work.<br>

### Reference
* "Count Based Exploration with Successor Representation", Machaldo et al, 2018<br>
https://arxiv.org/pdf/1807.11622.pdf <br>
