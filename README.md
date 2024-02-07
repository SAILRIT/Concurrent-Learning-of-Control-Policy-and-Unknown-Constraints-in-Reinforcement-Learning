# **Concurrent Learning of Control Policy and Unknown Constraints in Reinforcement Learning**

# **Concurrent Learning Framework**

This repository contains the official code accompanyong the paper [Concurrent Learning of Control Policy and Unknown Constraints in Reinforcement Learning](linktopaper). It implements a novel bilevel oprimization framework to concurrently synthesis safety constraints and safe control policies in environments with unknown safety constraints. It leverages [`Gymnasium`](https://github.com/Farama-Foundation/Gymnasium.git) and [`Safety_gymnasium`](https://github.com/PKU-Alignment/safety-gymnasium.git) environments, RL algorithms from [`Omnisafe`](https://github.com/PKU-Alignment/omnisafe.git) library,  a library for real-time monitoring of Signal Temporal Logic (STL), [`rtamt`](https://github.com/nickovic/rtamt.git) and [`GPyOpt` ](https://github.com/SheffieldML/GPyOpt.git) for Bayesian optimization. This project is built on Python 3.10 and utilizes a Conda environment for managing dependencies and ensuring compatibility across systems.

## Dependencies

The codebase depends on several key libraries, including modified versions of Safety Gymnasium and Omnisafe, tailored to meet the specific objectives of this research. The dependencies are as follows:

- [`rtamt`](https://github.com/nickovic/rtamt.git)
- [`GPyOpt` ](https://github.com/SheffieldML/GPyOpt.git)
- Modified [`Gymnasium`](https://github.com/Farama-Foundation/Gymnasium.git)
- Modified [`Safety_gymnasium`](https://github.com/PKU-Alignment/safety-gymnasium.git)
- Modified [`Omnisafe`](https://github.com/PKU-Alignment/omnisafe.git)

## Environment Setup

To replicate the environment and run the project successfully, follow the steps outlined below. These steps assume you have Conda installed on your system. If not, please install [Anaconda](https://www.anaconda.com/products/individual).

### Step 1: Create Conda Environment

Open your terminal and execute the following command to create a Conda environment named "Concurrent Learning Framework" with Python 3.10 and activate environemnet.

```terminal
conda create -name Concurrent_Learning_Framework python==3.10
conda activate Concurrent_Learning_Framework
```

### Step 2: Install Dependencies

With the environment activated, install the required libraries using pip. Here's how to install each:

#### rtamt

To install `rtamt`, run the following command.

```terminal
pip install "git+rtamt git repository URL>](https://github.com/nickovic/rtamt.git"
```

#### GPyOpt and GPy

`GPyOpt` can be installed directly via pip, which will also install `GPy` as a dependency:

```terminal
pip install GPyOpt
```

#### Safety Gymnasium and Omnisafe

For the modified versions of Safety Gymnasium and Omnisafe, we first install the currently available versions of these packages as follows: 
```terminal
pip install safety_gymansium
pip install omnisafe
```

then replace the `Omnisafe`, `Safety_gymnasium`, and `Gymnasium` packages installed by the modified versions provided in the zipped files given in this repository. The extracted files should be placed in the "Site-packages" of your Conda environment to ensure proper integration. 
 
## Running the Project

To run the project, navigate to the project's root directory followed by the case study you are interested in running and `run<nameofcasestudy>.py` in the appropriate directory. For example, to run Safe Navigation Circle (SNC) Experiment, in the appropriate directory, run:
```terminal
python runSNC.py
```
## License

[MIT License](LICENSE)

## Contact

For any queries regarding this project, please contact lay0005@mix.wvu.edu and we'll get back to you as soon as possible. 



