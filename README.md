# Python Music Visualization Framework
This framework is made to facilitate realtime music visualization for Caverage events.

## Setup
This project requires Python >=3.7, Linux, and PortAudio.

### Dev Setup
To install this framework for development run the following from the project directory:

1. Create a venv:
        python3 -m venv .venv

2. Activate venv:
        source .venv/bin/activate

3. Install dependencies:
        python -m pip install -e ".[tests]"

#### Editor Setup
Optimally your editor should be setup to run pylint, black, isort, and mypy when saving a file. You should ensure that the configuration is being pulled from the project directory (e.g. .pylintrc). Editors like Atom and VS Code's Python plugins will default to doing this, but you should ensure your editor behaves that way.

#### Running Tests
Tests can be ran with the following command, replacing $module_name with your module:

    pytest --cov=pymvf --cov_config=setup.cfg

Any files that begin with test_ in the tests will be automatically picked up by pytest. Running with the --cov flag will show the amount of code coverage for each file in the project.

You can also add the -x flag to stop the first failure, or --maxfail=$n to stop after N failures.

For more info on using pytest see: https://docs.pytest.org/en/stable/contents.html
