# seamlis

`seamlis` is a safe exploration and mapping framework. This repositry provides the implementation of the `seamlis` algorithm.


## Installation

To install `seamlis`, follow these steps:

1. Clone the repository:
   ```bash
   git --recursive clone https://github.com/tkkim-robot/seamlis.git
   cd seamlis
   ```

   If you've already cloned the repository without the --recursive flag, you can initialize and update the submodules with:
   ```bash
   git submodule update --init --recursive
   ```

2. (Optional) Create and activate a virtual environment:

3. Install the package and its dependencie:
   ```bash
   python -m pip install -e .
   ```
   Or, install packages manually (see [`setup.py`](https://github.com/tkkim-robot/safe_control/blob/main/setup.py)).

